"""Analog cohorts: the leaks that would invalidate the experiment, and the metric.

`docs/FINAL_SEGMENT_EXPERIMENT.md` only means anything if three things hold. The
signature is a function of the device's own past; a held-out building's outcomes
never reach the estimator that scores that building; and the cross-margin split
is really a partition of the pairs rather than a filter that quietly drops some.
Each is pinned here on the primitive rather than on a cached artifact, so the
suite runs without `outputs/`.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from tools.fj_segment import Lab, Standardiser, s1_scores, s2_scores
from tools.fj_signature import MIN_HISTORY, NAMES, signature_at


def _history(days: int = 600) -> tuple[np.ndarray, np.ndarray]:
    step = np.arange(days, dtype=float)
    voltage = 3.05 - 0.0004 * step + 0.004 * np.sin(step / 30.0)
    temperature = 20.0 + 5.0 * np.sin(step / 180.0)
    return voltage, temperature


class SignatureCausalityTest(unittest.TestCase):
    def test_nothing_at_or_after_the_cutoff_is_read(self) -> None:
        voltage, temperature = _history()
        before = signature_at(voltage, temperature, 400)
        spoiled_v, spoiled_t = voltage.copy(), temperature.copy()
        spoiled_v[400:] = 2.0
        spoiled_t[400:] = -50.0
        after = signature_at(spoiled_v, spoiled_t, 400)
        for name, a, b in zip(NAMES, before, after):
            self.assertTrue(
                (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9,
                msg=f"{name} read the future",
            )

    def test_too_little_history_is_nan_rather_than_a_guess(self) -> None:
        voltage, temperature = _history()
        out = signature_at(voltage, temperature, MIN_HISTORY - 1)
        self.assertTrue(all(np.isnan(v) for v in out))

    def test_the_signature_carries_no_identity(self) -> None:
        """No building, room, device or outcome may key a segment: test
        buildings are unseen, so a segment keyed on identity is worth nothing."""
        for name in NAMES:
            for banned in ("building", "room", "device", "eol", "lifetime", "due"):
                self.assertNotIn(banned, name.lower())

    def test_a_colder_device_gets_a_colder_signature(self) -> None:
        voltage, temperature = _history()
        warm = signature_at(voltage, temperature, 500)
        cold = signature_at(voltage, temperature - 8.0, 500)
        index = NAMES.index("t_mean_life")
        self.assertAlmostEqual(warm[index] - cold[index], 8.0, places=6)
        self.assertGreater(cold[NAMES.index("t_cold_frac")],
                           warm[NAMES.index("t_cold_frac")])


@dataclass
class _Frame:
    features: np.ndarray
    scenario: np.ndarray
    battery: np.ndarray
    building: np.ndarray
    probability: np.ndarray
    remaining: np.ndarray
    due: np.ndarray
    days_to_eol: np.ndarray
    substitute_eol: np.ndarray

    def column(self, name: str) -> np.ndarray:  # pragma: no cover - unused here
        raise NotImplementedError


def _synthetic(seed: int = 5, devices: int = 220, scenarios: int = 6):
    """Two buildings, every row's 42-day fate observed, margins well spread."""
    rng = np.random.default_rng(seed)
    battery = np.repeat([f"d{i:03d}" for i in range(devices)], scenarios)
    scenario = np.tile(np.arange(scenarios), devices)
    building = np.where(np.repeat(np.arange(devices) % 2, scenarios) == 0, "bA", "bB")
    rows = battery.size
    signature = rng.normal(size=(rows, len(NAMES)))
    for device in range(devices):  # a signature is a device property
        signature[device * scenarios:(device + 1) * scenarios] = signature[device * scenarios]
    margin = np.abs(rng.normal(0.06, 0.04, rows))
    base = np.clip(0.6 - 3.0 * margin + rng.normal(0, 0.05, rows), 0.01, 0.95)
    due = rng.random(rows) < base
    frame = _Frame(
        features=np.zeros((rows, 1)), scenario=scenario, battery=battery,
        building=building, probability=base,
        remaining=np.full(rows, 400.0), due=due,
        days_to_eol=np.where(due, 20.0, np.nan),
        substitute_eol=np.zeros(rows),
    )
    fold = (building == "bB").astype(int)
    ctx = {"frame": frame, "base": base, "margin": margin, "signature": signature,
           "fold": fold, "lab": Lab(frame, base, margin),
           "columns": list(range(len(NAMES)))}
    return ctx


class FoldDisciplineTest(unittest.TestCase):
    """A held-out building's own outcomes must not reach its own score."""

    def setUp(self) -> None:
        self.ctx = _synthetic()
        self.held = self.ctx["fold"] == 1

    def _tampered(self):
        """The same context with every held-out row's 42-day fate inverted."""
        frame = self.ctx["frame"]
        spoiled = frame.due.copy()
        spoiled[self.held] = ~spoiled[self.held]
        other = _Frame(**{**frame.__dict__, "due": spoiled,
                          "days_to_eol": np.where(spoiled, 20.0, np.nan)})
        ctx = dict(self.ctx)
        ctx["frame"] = other
        ctx["lab"] = Lab(other, self.ctx["base"], self.ctx["margin"])
        return ctx

    def test_s1_neighbour_outcomes_are_training_fold_only(self) -> None:
        honest, _ = s1_scores(self.ctx, neighbours=8, bandwidth=0.02)
        tampered, _ = s1_scores(self._tampered(), neighbours=8, bandwidth=0.02)
        mask = self.held & (self.ctx["lab"].mask)
        self.assertTrue(mask.any(), "the test needs held-out landmarks")
        # Guard against the assertion passing because nothing was scored at all:
        # `s1_scores` skips a fold whose training pool is too thin.
        self.assertTrue(np.any(honest[mask] != 0.0), "no held-out row was scored")
        np.testing.assert_allclose(honest[mask], tampered[mask], atol=1e-12)
        # ...and the control: the *other* building's score does move, because its
        # training pool is exactly the rows that were tampered with.
        other = (~self.held) & self.ctx["lab"].mask
        self.assertFalse(np.allclose(honest[other], tampered[other]))

    def test_s2_segment_offsets_are_training_fold_only(self) -> None:
        honest, _ = s2_scores(self.ctx, k=3, kappa=5.0)
        tampered, _ = s2_scores(self._tampered(), k=3, kappa=5.0)
        mask = self.held & self.ctx["lab"].mask
        np.testing.assert_allclose(honest[mask], tampered[mask], atol=1e-12)

    def test_standardiser_statistics_come_from_the_rows_it_was_given(self) -> None:
        values = np.column_stack([np.arange(100.0), np.arange(100.0) * 2])
        scaler = Standardiser(values[:50], np.ones(50))
        self.assertAlmostEqual(scaler.centre[0], np.median(values[:50, 0]))
        self.assertNotAlmostEqual(scaler.centre[0], np.median(values[:, 0]))


class MetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _synthetic(seed=9)
        self.lab = self.ctx["lab"]

    def test_cross_and_same_partition_the_pairs(self) -> None:
        split = self.lab.split
        self.assertEqual(int(split["cross"].sum()) + int(split["same"].sum()),
                         int(split["all"].sum()))
        self.assertFalse((split["cross"] & split["same"]).any())

    def test_a_perfect_and_a_reversed_ordering_bracket_the_scale(self) -> None:
        frame = self.lab.frame
        perfect = frame.due.astype(float)
        self.assertAlmostEqual(self.lab.concordance(perfect), 1.0)
        self.assertAlmostEqual(self.lab.concordance(-perfect), 0.0)
        self.assertAlmostEqual(self.lab.concordance(np.zeros(frame.due.size)), 0.5)

    def test_a_fold_is_scored_on_pairs_wholly_inside_it(self) -> None:
        """Scenarios mix buildings, so keying a fold on the due row alone would
        score a pair partly on training buildings."""
        fold = self.ctx["fold"]
        table = self.lab.report(self.ctx["base"], fold)
        for value in (0, 1):
            inside = ((fold[self.lab.positive] == value)
                      & (fold[self.lab.negative] == value))
            self.assertAlmostEqual(
                table[f"f{value}"],
                round(self.lab.concordance(self.ctx["base"], inside), 4))


if __name__ == "__main__":
    unittest.main()
