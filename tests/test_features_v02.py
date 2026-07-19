"""Tests for the v0.2 iteration: starvation-free confusion selector, hardened
mining (prescreen, LLM screen, quarantined gate), and the screen() API."""

from collections import Counter

from labelsmith import Config, TaskSpec, screen
from labelsmith.artifact import label_component
from labelsmith.data import Example
from labelsmith.features.confusion import ConfusionComponentSelector, ConfusionState
from labelsmith.features.mining import llm_screen, looks_like_label_noise, select_exemplars
from labelsmith.program import Prediction
from labelsmith.testing import fake_lms_for, patch_lms

LABELS = ["a", "b", "c", "d", "e", "f"]


def pred(text, gold, predicted):
    correct = gold == predicted
    return Prediction(text=text, gold=gold, predicted=predicted, raw=predicted,
                      correct=correct, feedback="")


def state_with_pairs(pairs, rounds=1):
    """pairs: list of (gold, predicted, count) -> a ConfusionState whose
    last_preds contain that many errors per pair."""
    st = ConfusionState()
    preds = []
    i = 0
    for gold, wrong, count in pairs:
        for _ in range(count):
            preds.append(pred(f"t{i}", gold, wrong))
            i += 1
    for _ in range(rounds):
        st.update(preds)
    return st


class TestSelector:
    def make_candidate(self):
        cand = {label_component(name): f"def {name}" for name in LABELS}
        cand["task_instruction"] = "x"
        cand["boundary_rules"] = "y"
        return cand

    def collect_selections(self, st, n_calls=40):
        sel = ConfusionComponentSelector(st, list(self.make_candidate()))
        cand = self.make_candidate()
        picks = []
        for _ in range(n_calls):
            picks.append(tuple(sel(None, [], [], 0, cand)))
        return picks

    def test_no_pair_starves(self):
        st = state_with_pairs([("a", "b", 24), ("c", "d", 3), ("e", "f", 2)])
        picks = self.collect_selections(st, 40)
        flat = Counter(c for p in picks for c in p)
        # every confused label gets updates — including the rare pairs
        for name in LABELS:
            assert flat[label_component(name)] > 0, f"{name} starved: {flat}"
        # the heavy pair still gets the most attention
        assert flat[label_component("a")] > flat[label_component("c")]
        assert flat[label_component("a")] > flat[label_component("e")]

    def test_proportionality(self):
        st = state_with_pairs([("a", "b", 20), ("c", "d", 10)])
        picks = self.collect_selections(st, 30)
        pair_picks = Counter(p for p in picks if p and p[0].startswith("label::"))
        ab = pair_picks[(label_component("a"), label_component("b"))]
        cd = pair_picks[(label_component("c"), label_component("d"))]
        assert ab > cd  # 2:1 weights -> roughly 2:1 schedule
        assert cd >= 5  # but the lighter pair is served regularly

    def test_shared_components_still_visited(self):
        st = state_with_pairs([("a", "b", 10)])
        picks = self.collect_selections(st, 16)
        flat = {c for p in picks for c in p}
        assert "boundary_rules" in flat and "task_instruction" in flat

    def test_round_robin_before_confusion_data(self):
        sel = ConfusionComponentSelector(ConfusionState(), list(self.make_candidate()))
        picks = [tuple(sel(None, [], [], 0, self.make_candidate())) for _ in range(16)]
        flat = {c for p in picks for c in p}
        assert len(flat) >= len(LABELS)  # cycles everything


class TestMiningScreens:
    def test_consistency_prescreen_flags_never_wavering(self):
        st = ConfusionState()
        preds_noise = [pred("noisy", "a", "b")]
        preds_hard = [pred("hard", "c", "d")]
        for i in range(6):
            st.update(preds_noise + (preds_hard if i < 3 else [pred("hard", "c", "c")]))
        assert looks_like_label_noise(st, "noisy")  # wrong every round, same way
        assert not looks_like_label_noise(st, "hard")  # fixed by later candidates

    def test_select_exemplars_excludes_noise(self):
        st = ConfusionState()
        finals = [pred("noisy", "a", "b"), pred("hard", "c", "d")]
        for _ in range(6):
            st.update(finals)
        # both persistently missed; "noisy" == "hard" profile here, so relax:
        # make hard waver between two wrong labels
        st2 = ConfusionState()
        for i in range(6):
            st2.update([pred("noisy", "a", "b"),
                        pred("hard", "c", "d" if i % 2 else "e")])
        chosen = select_exemplars(st2, [pred("noisy", "a", "b"), pred("hard", "c", "d")])
        texts = {e["text"] for e in chosen}
        assert "noisy" not in texts
        assert "hard" in texts

    def test_llm_screen_parses_keep_list(self):
        exemplars = [{"text": "t0", "label": "a"}, {"text": "t1", "label": "b"}]
        kept = llm_screen(exemplars, {"a": "letter a", "b": "letter b"}, lambda p: "[1]")
        assert kept == [exemplars[1]]

    def test_llm_screen_fails_open(self):
        exemplars = [{"text": "t0", "label": "a"}]
        assert llm_screen(exemplars, {}, lambda p: "garbage") == exemplars
        def boom(p):
            raise RuntimeError("lm down")
        assert llm_screen(exemplars, {}, boom) == exemplars


class TestScreen:
    def test_screen_verdicts(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        train = [Example(f"the {lab} signal {i}", lab) for lab in LABELS[:4] for i in range(30)]
        spec = TaskSpec.from_examples("scr", train)
        task_lm, refl_lm = fake_lms_for(train, [], spec.label_names)
        patch_lms(monkeypatch, task_lm, refl_lm)
        cfg = Config(task_model="fake/task", reflection_model="fake/reflection")
        res = screen(spec, train, cfg, sample_size=60)
        assert res.n == 60
        assert res.verdict in {"saturated", "marginal", "headroom"}
        # bare-taxonomy seed prompt + fake LM quality floor -> headroom expected
        assert res.verdict == "headroom"
        assert 0 < res.noise_floor < 0.1
        assert "headroom" in str(res).lower()

    def test_screen_uses_whole_small_dataset(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        train = [Example(f"x {lab} {i}", lab) for lab in LABELS[:2] for i in range(10)]
        spec = TaskSpec.from_examples("scr2", train)
        task_lm, refl_lm = fake_lms_for(train, [], spec.label_names)
        patch_lms(monkeypatch, task_lm, refl_lm)
        cfg = Config(task_model="fake/task", reflection_model="fake/reflection")
        assert screen(spec, train, cfg, sample_size=100).n == 20
