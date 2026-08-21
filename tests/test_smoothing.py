"""Pin bsai.smoothing against the official smooth_series, including the
incremental path that the production planner relies on."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from batteryswap_public.utils import smooth_series

from bsai.smoothing import SmoothingCache

DATASET = Path("dataset/train")
SAMPLE_DEVICES = 25


def _load_sample() -> pd.DataFrame:
    frame = pd.read_parquet(DATASET / "battery_metrics.parquet", engine="fastparquet")
    devices = sorted(frame["device_id"].unique())[:SAMPLE_DEVICES]
    frame = frame[frame["device_id"].isin(devices)]
    return frame.set_index(["device_id", "end_time"]).sort_index()


def _official(frame: pd.DataFrame) -> pd.DataFrame:
    out = smooth_series(frame).reset_index()
    out["end_time"] = pd.to_datetime(out["end_time"])
    return out


def _compare(cls, official: pd.DataFrame, ours: pd.DataFrame, note: str) -> None:
    merged = official.merge(
        ours, on=["device_id", "end_time"], how="outer", suffixes=("_off", "_ours")
    )
    cls.assertEqual(
        len(merged), len(official), f"{note}: our grid does not match the official grid"
    )
    for column in ("voltage", "temperature"):
        a = merged[f"{column}_off"].to_numpy(dtype=float)
        b = merged[f"{column}_ours"].to_numpy(dtype=float)
        cls.assertTrue(
            np.array_equal(np.isnan(a), np.isnan(b)),
            f"{note}: {column} missingness differs",
        )
        mask = ~np.isnan(a)
        np.testing.assert_allclose(
            a[mask], b[mask], rtol=0, atol=1e-12, err_msg=f"{note}: {column} differs"
        )


@unittest.skipUnless(
    (DATASET / "battery_metrics.parquet").exists(), "train dataset not available"
)
class SmoothingCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = _load_sample()
        cls.times = cls.raw.index.get_level_values("end_time")

    def test_single_pass_matches_official(self) -> None:
        cache = SmoothingCache()
        cache.update(self.raw)
        _compare(self, _official(self.raw), cache.frame(), "single pass")

    def test_incremental_matches_official(self) -> None:
        """Feed the cache in scenario-sized slices and require the same answer.

        The cuts land on Monday midnights, exactly like iterate_scenarios, so
        this also covers the partial-day boundary that forces the cache to
        re-read its last processed day.
        """
        end = self.times.max()
        cuts = [end - pd.Timedelta(days=k) for k in (56, 42, 28, 14, 0)]
        cache = SmoothingCache()
        for cut in cuts:
            cache.update(self.raw[self.times <= cut])
        _compare(self, _official(self.raw), cache.frame(), "incremental")

    def test_incremental_prefix_matches_truncated_official(self) -> None:
        """A cache stopped early must equal smoothing of the truncated input."""
        cut = self.times.max() - pd.Timedelta(days=30)
        truncated = self.raw[self.times <= cut]
        cache = SmoothingCache()
        cache.update(self.raw[self.times <= cut - pd.Timedelta(days=17)])
        cache.update(truncated)
        _compare(self, _official(truncated), cache.frame(), "prefix")


if __name__ == "__main__":
    unittest.main()
