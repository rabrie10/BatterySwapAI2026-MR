"""Template similarity at every scenario cutoff, out of fold by building.

For each of V8's five building folds, templates are built **only** from crossing
devices whose building sits in the other four, and every row from the held-out
buildings is scored against that bank. A held-out building's own death can
therefore never appear in a template that scores it, which is the one property
this experiment cannot afford to get wrong -- it is asserted in code and again
in ``tests/test_templates.py``.

The gate is within-scenario concordance on the same landmark population every
other candidate on this branch was measured on (top 40 by V8 probability per
scenario, censored rows excluded), where **V8 scores 0.7280**. The comparison of
interest is not only whether the template score beats that on its own, but
whether it adds anything to V8 -- and specifically whether it adds anything
*across* margins, which is where the matched-volatility signal could not act.

    python tools/fj_templates.py --report outputs/fj_templates.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.rerank import centred_rank  # noqa: E402
from bsai.templates import (  # noqa: E402
    WINDOWS,
    build_queries,
    build_templates,
    nearest_lead,
)
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402
from tools.fj_residual import v8_folds  # noqa: E402
from tools.fj_terminality import load_series  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
HORIZON = 42.0


def crossing_index(series: dict, dataset: Path) -> dict[str, int]:
    """First day the smoothed voltage is under 2.4 V, in each device's own grid."""
    table = pd.read_csv(dataset / "eol_times.csv")
    recorded = table[table["end_time"].notna()]
    stamps = {
        str(d): pd.Timestamp(t).normalize()
        for d, t in zip(recorded["device_id"], recorded["end_time"])
    }
    out: dict[str, int] = {}
    for device, (_voltage, _temperature, origin) in series.items():
        moment = stamps.get(str(device))
        if moment is None:
            continue
        out[str(device)] = int((moment - _EPOCH) / pd.Timedelta(days=1)) - origin
    return out


def cutoff_index(frame, series: dict, dataset: Path) -> np.ndarray:
    from batteryswap_public.utils import load_dataset

    _, _, _, scenarios = load_dataset(dataset)
    starts = np.asarray([
        int((pd.Timestamp(s["start_time"]).normalize() - _EPOCH) / pd.Timedelta(days=1))
        for s in scenarios
    ])
    out = np.full(frame.scenario.size, -1, dtype=int)
    for row in range(frame.scenario.size):
        entry = series.get(str(frame.battery[row]))
        if entry is None:
            continue
        index = starts[frame.scenario[row]] - entry[2]
        out[row] = min(max(index, -1), entry[0].size - 1)
    return out


def concordance(frame, score: np.ndarray, mask: np.ndarray) -> float:
    good = ties = total = 0.0
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero((frame.scenario == index) & mask)
        y = frame.due[rows]
        if y.sum() == 0 or y.all():
            continue
        gap = score[rows][y][:, None] - score[rows][~y][None, :]
        good += (gap > 0).sum()
        ties += (gap == 0).sum()
        total += gap.size
    return (good + 0.5 * ties) / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_template_scores.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_templates.json"))
    args = parser.parse_args()

    started = time.time()
    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    series = load_series(args.series)
    crossing = crossing_index(series, args.dataset)
    cutoff = cutoff_index(frame, series, args.dataset)
    print(f"{len(crossing)} crossing devices, "
          f"{np.unique(frame.battery[frame.due]).size} of them ever due in a scenario")

    devices = pd.read_csv(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"].astype(str),
                           devices["building_id"].astype(str)))
    fold_of = v8_folds(args.folds)
    row_fold = np.asarray([fold_of.get(b, -1) for b in frame.building])
    folds = sorted(set(fold_of.values()))

    # Landmarks: the same population every other candidate was scored on.
    observed = np.isfinite(frame.days_to_eol) | (frame.remaining >= HORIZON)
    usable = frame.due | observed
    candidates = np.zeros(base.size, dtype=bool)
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        candidates[rows[np.argsort(-base[rows], kind="stable")][: args.candidates]] = True
    landmarks = usable & candidates
    print(f"landmarks {int(landmarks.sum())}, V8 concordance "
          f"{concordance(frame, base, landmarks):.4f}")
    print()

    scores: dict[str, np.ndarray] = {}
    for mode in ("anchored", "level"):
        for channel in ("voltage", "adjusted"):
            for width in WINDOWS:
                lead = np.full(base.size, np.nan)
                distance = np.full(base.size, np.nan)
                bank_sizes = []
                for fold in folds:
                    held = row_fold == fold
                    held_buildings = {
                        b for b, f in
                        zip(frame.building, row_fold) if f == fold
                    }
                    allowed = {
                        d for d in crossing
                        if building_of.get(d, "") not in held_buildings
                    }
                    # The invariant, asserted rather than assumed.
                    assert not any(
                        building_of.get(d, "") in held_buildings for d in allowed
                    )
                    bank = build_templates(
                        series, crossing, allowed,
                        width=width, channel=channel, mode=mode,
                    )
                    bank_sizes.append(len(bank))
                    rows = np.flatnonzero(held & (cutoff >= 0))
                    if rows.size == 0 or len(bank) == 0:
                        continue
                    queries, ok = build_queries(
                        series, frame.battery[rows], cutoff[rows],
                        width=width, channel=channel, mode=mode,
                    )
                    predicted, closest = nearest_lead(queries[ok], bank, k=args.k)
                    lead[rows[ok]] = predicted
                    distance[rows[ok]] = closest
                name = f"{mode}_{channel}_{width}"
                scores[f"lead_{name}"] = lead
                scores[f"dist_{name}"] = distance
                covered = np.isfinite(lead[landmarks]).mean()
                # Risk is *short* predicted lead, so the score is negated.
                value = concordance(frame, -np.nan_to_num(lead, nan=1e9), landmarks)
                print(f"  {name:>22}  banks {int(np.mean(bank_sizes)):5d}  "
                      f"coverage {covered:5.1%}  concordance {value:.4f}  "
                      f"({time.time() - started:.0f}s)", flush=True)

    np.savez_compressed(args.out, **scores)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "v8_concordance": round(concordance(frame, base, landmarks), 4),
        "landmarks": int(landmarks.sum()),
        "k": args.k,
        "alone": {
            name.replace("lead_", ""): round(
                concordance(frame, -np.nan_to_num(value, nan=1e9), landmarks), 4
            )
            for name, value in scores.items() if name.startswith("lead_")
        },
    }, indent=1))
    print(f"\nwrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
