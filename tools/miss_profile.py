"""What do the invisible deaths look like?

``tools/error_profile.py`` asks what separates a false positive from a true
positive. This asks the opposite and much more expensive question. Measured on
the ledger, the due batteries the plan misses carry a median predicted
probability of **0.008** -- the model does not merely rank them low, it says they
are safe. There are 3.9 of them per scenario and they are the single largest
term in the score (``late_swap`` 1026 out of 2145 out of fold).

Only 3.2% of those misses were ranked above the lowest-ranked battery the
planner did swap, so this is not a decision-layer problem. It is a population
the forecast cannot see at all.

So the question this answers is narrow: **inside the population the model calls
safe, what separates the devices that die within six weeks from the ones that do
not?** If nothing does, the failures are genuinely unforecastable from this data
and recall is a dead end. If something does, it is worth several hundred points,
because catching them lets the planner swap fewer batteries rather than more.

Cutoffs are the 48 scenario start dates, not the strided training grid -- a
scenario asks about every alive device on one date, and those two populations
are not interchangeable (``HANDOVER.md`` trap 2).

    python tools/miss_profile.py --folds outputs/v7_folds.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices, load_scenarios

from bsai.features import FEATURE_NAMES
from bsai.hazard import build_training_frame
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--safe-threshold", type=float, default=0.10)
    parser.add_argument("--report", type=Path, default=Path("outputs/v9_miss_profile.json"))
    parser.add_argument("--frame-out", type=Path, default=Path("outputs/v9_scenario_frame.npz"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    observation_end = devices.set_index("device_id")["end_time"]
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"],
        format="ISO8601",
    ).dt.tz_localize(None)
    scenarios = load_scenarios(args.dataset / "scenarios.json")
    cutoff_days = np.array(
        [_ordinal(s["start_time"]) for s in scenarios], dtype=np.int64
    )

    print("smoothing and within-day shape...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw

    eol_index: dict[str, int | None] = {}
    observation_index: dict[str, int] = {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = (
            None if pd.isna(moment) else _ordinal(moment) - series.origin
        )
        end = observation_end.get(device_id)
        observation_index[device_id] = (
            (series.origin + len(series) - 1)
            if pd.isna(end)
            else _ordinal(end) - series.origin
        )

    print("building the scenario-cutoff frame...", flush=True)
    frame = build_training_frame(
        cache, eol_index, building_of, observation_index,
        shape_cache=shape_cache, cutoff_days=cutoff_days,
    )
    due = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & ((frame.crossing - frame.cutoff) <= DECISION_HORIZON)
        & (frame.crossing <= frame.observation_end)
    )
    remaining = np.maximum(frame.observation_end - frame.cutoff, 0).astype(float)
    days_to_eol = np.where(frame.crossing >= 0, frame.crossing - frame.cutoff, 10**6)
    print(f"  {len(frame)} device-scenarios, {int(due.sum())} due "
          f"({due.mean():.4f})", flush=True)

    print("scoring out of fold...", flush=True)
    bundle = joblib.load(args.folds)
    probability = np.zeros(len(frame))
    buildings = np.asarray([str(b) for b in frame.building])
    for building in np.unique(buildings):
        model = bundle["by_building"][building]
        model.volatility_scale = 1.0
        mask = buildings == building
        grid = model.predict_grid(frame.features[mask], remaining[mask])
        column = list(model.horizons).index(DECISION_HORIZON)
        probability[mask] = grid[:, column]

    np.savez_compressed(
        args.frame_out,
        features=frame.features, due=due, probability=probability,
        remaining=remaining, days_to_eol=days_to_eol,
        device=frame.device, building=frame.building, cutoff=frame.cutoff,
    )

    safe = probability < args.safe_threshold
    invisible = safe & due
    print()
    print(f"=== the population the model calls safe (p < {args.safe_threshold}) ===")
    print(f"  rows                {int(safe.sum()):7d}  ({safe.mean():.1%} of all)")
    print(f"  due inside it       {int(invisible.sum()):7d}  "
          f"({invisible.sum()/max(due.sum(),1):.1%} of ALL due batteries)")
    print(f"  due rate inside it  {due[safe].mean():.5f}   "
          f"(overall {due.mean():.5f}, visible {due[~safe].mean():.5f})")
    print(f"  days to EOL among the invisible ones: "
          f"median {np.median(days_to_eol[invisible]):.0f}, "
          f"q10 {np.quantile(days_to_eol[invisible],0.1):.0f}, "
          f"q90 {np.quantile(days_to_eol[invisible],0.9):.0f}")

    # Within the safe population: which feature separates the deaths?
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        column = frame.features[safe, index].astype(float)
        finite = np.isfinite(column)
        if finite.sum() < 500:
            continue
        target = due[safe][finite]
        if target.sum() < 20 or target.sum() == finite.sum():
            continue
        auc = float(roc_auc_score(target, column[finite]))
        pos, neg = column[finite][target == 1], column[finite][target == 0]
        rows.append(
            {
                "feature": name,
                "auc": round(max(auc, 1 - auc), 4),
                "direction": "higher" if auc > 0.5 else "lower",
                "median_due": round(float(np.median(pos)), 6),
                "median_not_due": round(float(np.median(neg)), 6),
                "n": int(finite.sum()),
            }
        )
    table = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print()
    print(f"=== separating the {int(invisible.sum())} invisible deaths from the rest "
          f"of the safe population, one feature at a time ===")
    print(table.head(20).to_string(index=False))

    # The control that matters: is this just the model under-using a feature it
    # already has, or is the safe population genuinely unseparable?
    print()
    print("=== control: the same ranking on the VISIBLE population ===")
    visible_rows = []
    for index, name in enumerate(FEATURE_NAMES[:0]):
        pass
    best = table.head(8).feature.tolist()
    for name in best:
        index = FEATURE_NAMES.index(name)
        column = frame.features[~safe, index].astype(float)
        finite = np.isfinite(column)
        target = due[~safe][finite]
        if target.sum() < 10 or target.sum() == finite.sum():
            continue
        auc = float(roc_auc_score(target, column[finite]))
        visible_rows.append(
            {"feature": name, "auc_visible": round(max(auc, 1 - auc), 4)}
        )
    if visible_rows:
        print(pd.DataFrame(visible_rows).to_string(index=False))

    summary = {
        "safe_threshold": args.safe_threshold,
        "n_rows": int(len(frame)),
        "n_due": int(due.sum()),
        "n_safe": int(safe.sum()),
        "n_invisible_deaths": int(invisible.sum()),
        "share_of_due_that_is_invisible": round(
            float(invisible.sum() / max(due.sum(), 1)), 4
        ),
        "due_rate_in_safe": round(float(due[safe].mean()), 5),
        "separators": table.head(25).to_dict("records"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.report} and {args.frame_out}")


if __name__ == "__main__":
    main()
