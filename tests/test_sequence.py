"""The submission contract for the sequence remap.

What has to be true for `models/sequence_tcn.json` to be shippable, asserted
rather than assumed:

* the NumPy forward pass is the model that was measured, not a lookalike;
* the artifact cannot silently be a Git-LFS pointer, which is how this project
  once degraded a whole submission (`script.py::_is_lfs_pointer`);
* a window is a function of the past only;
* the deployment is **order-only** -- V8's own per-scenario CDF multiset comes
  back out, in a different order, carrying identical risk mass;
* a row the model cannot score keeps V8's ordering rather than falling to noise;
* the same input gives the same output, every time, with no seed anywhere.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from bsai.hazard import HORIZON_GRID
from bsai.rerank import RankRemapModel
from bsai.sequence import (
    CHANNELS,
    HISTORY,
    HORIZONS,
    QUANTILES,
    LiveSequenceScorer,
    SequenceModel,
    _Filled,
    crossing_probability,
    window_at,
)

ARTIFACT = Path("models/sequence_tcn.json")


def _series(days: int = 400, gaps: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    step = np.arange(days, dtype=float)
    voltage = 3.05 - 0.0005 * step
    temperature = 20.0 + 4.0 * np.sin(step / 90.0)
    for start, stop in gaps:
        voltage[start:stop] = np.nan
    return voltage, temperature


class _FakeSeries:
    def __init__(self, voltage, temperature, origin=0):
        self.smooth_voltage = voltage
        self.smooth_temperature = temperature
        self.origin = origin

    def __len__(self):
        return self.smooth_voltage.size

    def index_of(self, ordinal):
        return int(ordinal) - self.origin


class _FakeCache:
    def __init__(self, devices):
        self.devices = devices


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest(f"{ARTIFACT} not built")

    def test_it_loads_and_declares_what_it_holds(self) -> None:
        model = SequenceModel.load(ARTIFACT)
        self.assertEqual(len(model.folds), 5)
        counted = sum(t.size for t in model.folds[0].tensors.values())
        self.assertEqual(counted, 12899)

    def test_a_git_lfs_pointer_raises_instead_of_being_parsed(self) -> None:
        pointer = Path(self.enterContext(_temp_dir())) / "sequence_tcn.json"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 12\n")
        with self.assertRaises(OSError) as caught:
            SequenceModel.load(pointer)
        self.assertIn("Git-LFS pointer", str(caught.exception))

    def test_a_declared_parameter_count_that_disagrees_raises(self) -> None:
        payload = json.loads(ARTIFACT.read_text())
        payload["parameters_per_fold"] = 1
        broken = Path(self.enterContext(_temp_dir())) / "broken.json"
        broken.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            SequenceModel.load(broken)

    def test_the_artifact_is_not_tracked_by_git_lfs(self) -> None:
        """`.gitattributes` sends *.pt, *.npz and *.npy through LFS. Not this."""
        attributes = Path(".gitattributes").read_text()
        for pattern in ("*.json filter=lfs", "models/** filter=lfs"):
            self.assertNotIn(pattern, attributes)


class ForwardPassTest(unittest.TestCase):
    def setUp(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest(f"{ARTIFACT} not built")
        self.model = SequenceModel.load(ARTIFACT)

    def test_numpy_matches_torch(self) -> None:
        try:
            import torch

            from tools.fj_tcn import make_model
        except ImportError:
            self.skipTest("torch not installed")
        source = Path("outputs/fj_tcn.pt")
        if not source.exists():
            self.skipTest("training artifact not present")
        payload = torch.load(source, weights_only=False)
        TCN = make_model(torch)
        rng = np.random.default_rng(2)
        windows = rng.normal(size=(4, CHANNELS, HISTORY))
        for index, group in enumerate(sorted(payload["states"])):
            net = TCN(width=payload["width"])
            net.load_state_dict(payload["states"][group])
            net.eval()
            with torch.no_grad():
                expected = net(torch.from_numpy(windows.astype(np.float32))).numpy()
            got = self.model._forward_one(windows, self.model.folds[index])
            self.assertLess(float(np.abs(expected - got).max()), 1e-5)

    def test_it_is_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        windows = rng.normal(size=(6, CHANNELS, HISTORY))
        temperature = rng.normal(20.0, 4.0, size=(6, HISTORY))
        first = self.model.predict(windows, temperature)
        second = self.model.predict(windows, temperature)
        np.testing.assert_array_equal(first, second)

    def test_quantiles_are_ordered_and_the_tail_never_saturates(self) -> None:
        quantiles = np.array([[-0.20, -0.15, -0.06, -0.01, 0.01, 0.03, 0.05]])
        previous = -1.0
        for threshold in (-1.0, -0.5, -0.3, -0.05, 0.0, 0.5):
            value = float(crossing_probability(quantiles, np.array([threshold]))[0])
            self.assertGreater(value, previous)
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)
            previous = value


class WindowTest(unittest.TestCase):
    def test_a_window_reads_only_its_own_past(self) -> None:
        voltage, temperature = _series()
        state = _Filled(voltage, temperature)
        before = window_at(state, 300)
        spoiled_v, spoiled_t = voltage.copy(), temperature.copy()
        spoiled_v[301:] = -99.0
        spoiled_t[301:] = -99.0
        after = window_at(_Filled(spoiled_v, spoiled_t), 300)
        np.testing.assert_allclose(before[0], after[0], atol=1e-12)
        np.testing.assert_allclose(before[1], after[1], atol=1e-12)

    def test_shape_and_short_history(self) -> None:
        voltage, temperature = _series()
        state = _Filled(voltage, temperature)
        self.assertEqual(window_at(state, 200)[0].shape, (CHANNELS, HISTORY))
        self.assertIsNone(window_at(state, HISTORY - 2))
        self.assertIsNone(window_at(state, 10_000))

    def test_the_mask_channel_marks_gaps_and_the_fill_is_causal(self) -> None:
        voltage, temperature = _series(gaps=((250, 260),))
        state = _Filled(voltage, temperature)
        built = window_at(state, 265)
        self.assertAlmostEqual(float(built[0][3].sum()), HISTORY - 10)
        # the filled value inside a gap is the last observed one before it
        self.assertAlmostEqual(state.filled[255], voltage[249], places=9)


class OrderOnlyTest(unittest.TestCase):
    """The invariant the whole deployment rests on."""

    def setUp(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest(f"{ARTIFACT} not built")
        rng = np.random.default_rng(7)
        self.count = 24
        base = np.sort(rng.uniform(0.01, 0.9, self.count))[::-1]
        self.grid = np.clip(
            base[:, None] * np.linspace(0.2, 1.0, len(HORIZON_GRID))[None, :], 0, 1)
        self.remaining = np.full(self.count, 300.0)
        self.features = np.zeros((self.count, 64), dtype=np.float32)
        self.features[:, 0] = rng.uniform(2.42, 2.9, self.count)
        voltage, temperature = _series(days=500)
        devices = {}
        for index in range(self.count):
            shifted = voltage - 0.0002 * index * np.arange(voltage.size)
            devices[f"d{index}"] = _FakeSeries(shifted, temperature)
        self.cache = _FakeCache(devices)
        self.devices = np.asarray([f"d{i}" for i in range(self.count)])

    def _remapped(self, weight: float) -> np.ndarray:
        scorer = LiveSequenceScorer(model=SequenceModel.load(ARTIFACT), weight=weight)
        scorer.bind(self.cache, 400)
        model = RankRemapModel(base=_Passthrough(self.grid), scorer=scorer)
        return model.predict_grid(self.features, self.remaining, self.devices), scorer

    def test_the_multiset_and_the_risk_mass_are_preserved(self) -> None:
        out, scorer = self._remapped(1.0)
        self.assertGreater(scorer.scored, 0, "nothing was scored; test is vacuous")
        column = list(HORIZON_GRID).index(42)
        np.testing.assert_allclose(
            np.sort(out[:, column]), np.sort(self.grid[:, column]), atol=1e-12)
        np.testing.assert_allclose(out[:, column].sum(), self.grid[:, column].sum(),
                                   atol=1e-9)
        # the whole curve travels together, not just the decision column
        np.testing.assert_allclose(np.sort(out, axis=0), np.sort(self.grid, axis=0),
                                   atol=1e-12)

    def test_it_actually_reorders(self) -> None:
        out, _ = self._remapped(1.0)
        column = list(HORIZON_GRID).index(42)
        self.assertFalse(np.allclose(out[:, column], self.grid[:, column]),
                         "the remap changed nothing; the scorer is inert")

    def test_weight_zero_is_exactly_the_incumbent(self) -> None:
        out, _ = self._remapped(0.0)
        np.testing.assert_allclose(out, self.grid, atol=1e-12)

    def test_an_unscoreable_row_keeps_the_incumbent_order(self) -> None:
        scorer = LiveSequenceScorer(model=SequenceModel.load(ARTIFACT), weight=1.0)
        scorer.bind(_FakeCache({}), 400)
        model = RankRemapModel(base=_Passthrough(self.grid), scorer=scorer)
        out = model.predict_grid(self.features, self.remaining, self.devices)
        np.testing.assert_allclose(out, self.grid, atol=1e-12)
        self.assertEqual(scorer.scored, 0)

    def test_repeated_calls_agree(self) -> None:
        first, _ = self._remapped(1.0)
        second, _ = self._remapped(1.0)
        np.testing.assert_array_equal(first, second)


class _Passthrough:
    """A minimal base model returning a fixed grid, for the remap tests."""

    horizons = HORIZON_GRID
    model_version = "test/passthrough"

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid

    def predict_grid(self, features, remaining, devices=None):
        return self.grid.copy()


def _temp_dir():
    import tempfile

    return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
