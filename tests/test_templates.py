"""EOL-aligned templates: the leak that would invalidate the experiment, and the shapes.

The one property this experiment cannot afford to get wrong is that no crossing
device from a held-out building contributes a template to a query from that
building. It is asserted inside ``tools/fj_templates.py`` on every fold and
tested here on the primitive.
"""

from __future__ import annotations

import unittest

import numpy as np

from bsai.templates import (
    MAX_MISSING,
    TemplateBank,
    adjusted,
    build_queries,
    build_templates,
    nearest_lead,
    normalise,
)


def _series(n: int = 500, drop_at: int = 400) -> tuple[np.ndarray, np.ndarray]:
    voltage = np.full(n, 2.9)
    voltage[drop_at:] = 2.9 - np.linspace(0.0, 0.6, n - drop_at)
    temperature = np.full(n, 20.0)
    return voltage, temperature


class FoldDisciplineTest(unittest.TestCase):
    def setUp(self) -> None:
        voltage, temperature = _series()
        self.series = {
            "train_a": (voltage, temperature, 0),
            "train_b": (voltage, temperature, 0),
            "held": (voltage, temperature, 0),
        }
        self.crossing = {"train_a": 450, "train_b": 460, "held": 455}

    def test_only_allowed_devices_contribute(self) -> None:
        bank = build_templates(
            self.series, self.crossing, {"train_a", "train_b"},
            width=30, channel="voltage", mode="anchored",
        )
        self.assertGreater(len(bank), 0)
        self.assertNotIn("held", set(bank.device.tolist()))
        self.assertEqual({"train_a", "train_b"}, set(bank.device.tolist()))

    def test_an_empty_allowance_yields_an_empty_bank(self) -> None:
        bank = build_templates(
            self.series, self.crossing, set(),
            width=30, channel="voltage", mode="anchored",
        )
        self.assertEqual(len(bank), 0)

    def test_a_non_crossing_device_contributes_nothing(self) -> None:
        bank = build_templates(
            self.series, {"train_a": -1}, {"train_a"},
            width=30, channel="voltage", mode="anchored",
        )
        self.assertEqual(len(bank), 0)

    def test_templates_never_read_past_the_crossing(self) -> None:
        """Every segment ends at or before its device's crossing day."""
        bank = build_templates(
            self.series, self.crossing, {"train_a"},
            width=20, channel="voltage", mode="level",
        )
        self.assertTrue((bank.lead >= 0).all())


class ShapeTest(unittest.TestCase):
    def test_anchored_removes_the_level_and_level_keeps_it(self) -> None:
        window = np.array([2.9, 2.8, 2.7])
        self.assertTrue(np.allclose(normalise(window, "anchored"), [0.2, 0.1, 0.0]))
        self.assertTrue(np.allclose(normalise(window, "level"), window))

    def test_two_windows_differing_only_by_offset_are_identical_when_anchored(self) -> None:
        a = np.array([2.90, 2.85, 2.80])
        self.assertTrue(
            np.allclose(normalise(a, "anchored"), normalise(a - 0.4, "anchored"))
        )
        self.assertFalse(np.allclose(normalise(a, "level"), normalise(a - 0.4, "level")))

    def test_temperature_adjustment_removes_a_pure_thermal_swing(self) -> None:
        temperature = 20.0 + 5.0 * np.sin(np.arange(200) / 30.0)
        voltage = 2.7 + 0.00463 * (temperature - 20.0)
        corrected = adjusted(voltage, temperature)
        self.assertLess(float(corrected.std()), 1e-9)

    def test_a_gappy_window_is_rejected(self) -> None:
        voltage = np.full(300, 2.8)
        voltage[200:260] = np.nan  # more than MAX_MISSING of a 60-day window
        queries, usable = build_queries(
            {"d": (voltage, np.full(300, 20.0), 0)}, np.asarray(["d"]),
            np.asarray([259]), width=60, channel="voltage", mode="level",
        )
        self.assertGreater(0.5, MAX_MISSING)
        self.assertFalse(usable[0])


class NearestLeadTest(unittest.TestCase):
    def test_an_exact_match_recovers_its_own_lead(self) -> None:
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(40, 12)).astype(np.float32) * 5.0
        lead = np.arange(40) * 7
        bank = TemplateBank(vectors, lead, np.asarray([f"d{i}" for i in range(40)]))
        predicted, closest = nearest_lead(vectors[[3, 17]].astype(np.float64), bank, k=1)
        self.assertAlmostEqual(predicted[0], 21.0, places=6)
        self.assertAlmostEqual(predicted[1], 119.0, places=6)
        self.assertAlmostEqual(closest[0], 0.0, places=5)

    def test_an_empty_bank_returns_nan(self) -> None:
        bank = TemplateBank(np.zeros((0, 5)), np.zeros(0, int), np.zeros(0, object))
        predicted, closest = nearest_lead(np.zeros((3, 5)), bank)
        self.assertTrue(np.isnan(predicted).all())
        self.assertTrue(np.isnan(closest).all())

    def test_the_estimate_stays_inside_the_lead_range(self) -> None:
        rng = np.random.default_rng(4)
        vectors = rng.normal(size=(60, 8)).astype(np.float32)
        lead = np.arange(60) * 5
        bank = TemplateBank(vectors, lead, np.asarray([f"d{i}" for i in range(60)]))
        predicted, _ = nearest_lead(rng.normal(size=(25, 8)), bank, k=10)
        self.assertTrue((predicted >= lead.min()).all())
        self.assertTrue((predicted <= lead.max()).all())


if __name__ == "__main__":
    unittest.main()
