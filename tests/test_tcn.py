"""The sequence model's corpus: causality, censoring, and device boundaries.

A TCN trained on windows is only worth reading if a window at day `t` is a
function of `[t - HISTORY + 1, t]` and nothing else, if a horizon whose future
day was never observed is dropped rather than imputed, and if no window straddles
two devices. Each is pinned on the primitive, so the suite runs without a trained
model or the cached series.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.fj_tcn import (
    CHANNELS,
    HISTORY,
    HORIZONS,
    Corpus,
    _targets_through_eol,
    anchor_days,
    assert_origins_precede_eol,
)


def _device(days: int = 400, seed: int = 0, gaps: tuple[tuple[int, int], ...] = ()):
    step = np.arange(days, dtype=float)
    voltage = 3.05 - 0.0005 * step
    temperature = 20.0 + 4.0 * np.sin(step / 90.0)
    for start, stop in gaps:
        voltage[start:stop] = np.nan
    return voltage, temperature


def _series(n: int = 4, **kwargs):
    return {f"d{i}": (*_device(seed=i, **kwargs), 0) for i in range(n)}


class CorpusCausalityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.series = _series()
        self.corpus = Corpus(self.series, stride=7)

    def test_a_window_reads_only_its_own_past(self) -> None:
        rows = np.array([5, 11, 20])
        before = self.corpus.batch(rows, 20.0, 4.0)
        spoiled = {}
        for name, (voltage, temperature, origin) in self.series.items():
            voltage, temperature = voltage.copy(), temperature.copy()
            spoiled[name] = (voltage, temperature, origin)
        # Corrupt every day strictly after each queried anchor.
        latest = int(self.corpus.anchor[rows].max())
        offset = self.corpus.offset[self.corpus.owner[rows.max()]]
        for name, (voltage, temperature, origin) in spoiled.items():
            voltage[latest - offset + 1:] = -99.0
            temperature[latest - offset + 1:] = -99.0
        after = Corpus(spoiled, stride=7).batch(rows, 20.0, 4.0)
        np.testing.assert_allclose(before, after, atol=1e-12)

    def test_the_window_is_exactly_history_long(self) -> None:
        batch = self.corpus.batch(np.array([3]), 20.0, 4.0)
        self.assertEqual(batch.shape, (1, CHANNELS, HISTORY))

    def test_no_window_straddles_two_devices(self) -> None:
        start = self.corpus.anchor - HISTORY + 1
        owner_start = self.corpus.offset[self.corpus.owner]
        self.assertTrue(np.all(start >= owner_start))
        end = self.corpus.offset[self.corpus.owner] + self.corpus.length[self.corpus.owner]
        self.assertTrue(np.all(self.corpus.anchor < end))

    def test_forward_fill_never_reaches_forward(self) -> None:
        series = _series(n=1, gaps=((200, 230),))
        corpus = Corpus(series, stride=1)
        voltage = series["d0"][0]
        for day in (199, 205, 229, 230, 260):
            observed = np.flatnonzero(np.isfinite(voltage[:day + 1]))
            self.assertAlmostEqual(corpus.filled[day], voltage[observed[-1]], places=9)


class CensoringTest(unittest.TestCase):
    def test_an_unobserved_future_day_is_masked_not_imputed(self) -> None:
        series = _series(n=1, gaps=((300, 340),))
        corpus = Corpus(series, stride=1)
        voltage = series["d0"][0]
        for row in range(corpus.anchor.size):
            for column, horizon in enumerate(HORIZONS):
                future = int(corpus.anchor[row]) + horizon
                target = corpus.target[row, column]
                if future >= voltage.size or not np.isfinite(voltage[future]):
                    self.assertTrue(np.isnan(target),
                                    msg=f"horizon {horizon} at {row} was imputed")
                else:
                    self.assertFalse(np.isnan(target))

    def test_a_target_is_the_change_from_the_anchor(self) -> None:
        corpus = Corpus(_series(n=1), stride=11)
        voltage = _device()[0]
        row, column = 2, HORIZONS.index(42)
        anchor = int(corpus.anchor[row])
        self.assertAlmostEqual(
            corpus.target[row, column], voltage[anchor + 42] - voltage[anchor],
            places=9)

    def test_a_horizon_past_the_end_of_the_series_is_masked(self) -> None:
        corpus = Corpus(_series(n=1), stride=1)
        last = corpus.anchor.max()
        row = int(np.flatnonzero(corpus.anchor == last)[0])
        self.assertTrue(np.isnan(corpus.target[row, HORIZONS.index(42)]))


class OriginPrecedesEolTest(unittest.TestCase):
    """A window may *end* past EOL. It may never *start* there.

    A competition scenario only asks about an active battery: none of the 19,890
    cached rows sits at or after its device's crossing and the smallest margin
    any of them presents is 0.0000. Left unconstrained the corpus disagrees --
    2,869 of 109,481 windows anchor after EOL and 75.5 % of those sit below the
    2.4 V barrier, a regime inference never visits.
    """

    def setUp(self) -> None:
        self.series = _series(n=3)
        # d0 crosses at day 250; the other two are censored.
        self.stop = {"d0": 250}

    def test_no_anchor_reaches_its_devices_eol(self) -> None:
        corpus = Corpus(self.series, stride=1, stop=self.stop)
        day = anchor_days(corpus)
        owned = corpus.devices[corpus.owner]
        self.assertTrue(day[owned == "d0"].max() < 250)
        self.assertTrue(day[owned == "d1"].max() >= 250,
                        msg="a censored device must not be truncated")

    def test_the_assertion_fires_when_the_bound_is_dropped(self) -> None:
        unconstrained = Corpus(self.series, stride=1)
        with self.assertRaises(AssertionError):
            assert_origins_precede_eol(unconstrained, self.stop)
        assert_origins_precede_eol(Corpus(self.series, stride=1, stop=self.stop),
                                   self.stop)

    def test_targets_are_still_allowed_to_run_through_eol(self) -> None:
        """The terminal decline is the supervision, so it must survive."""
        corpus = Corpus(self.series, stride=1, stop=self.stop)
        through = _targets_through_eol(corpus, self.stop)
        self.assertGreater(int(through.sum()), 0)
        day = anchor_days(corpus)
        owned = corpus.devices[corpus.owner]
        late = np.flatnonzero((owned == "d0") & (day == 249))
        self.assertEqual(late.size, 1)
        column = HORIZONS.index(42)
        self.assertFalse(np.isnan(corpus.target[late[0], column]),
                         msg="a horizon crossing EOL was dropped")
        self.assertTrue(through[late[0]])

    def test_a_device_that_dies_before_the_window_length_contributes_nothing(self) -> None:
        corpus = Corpus(self.series, stride=1, stop={"d0": HISTORY - 5})
        owned = corpus.devices[corpus.owner]
        self.assertEqual(int((owned == "d0").sum()), 0)


class FoldDisciplineTest(unittest.TestCase):
    def test_a_training_pool_excludes_the_held_out_buildings_windows(self) -> None:
        """The one line `train` depends on, asserted rather than assumed."""
        corpus = Corpus(_series(n=6), stride=13)
        fold = np.array([0, 0, 1, 1, 2, 2])
        window_fold = fold[corpus.owner]
        for group in (0, 1, 2):
            rows = np.flatnonzero((window_fold != group) & (window_fold >= 0))
            self.assertTrue(rows.size > 0)
            self.assertNotIn(group, set(window_fold[rows].tolist()))
            devices = set(corpus.owner[rows].tolist())
            self.assertFalse(devices & set(np.flatnonzero(fold == group).tolist()))


if __name__ == "__main__":
    unittest.main()
