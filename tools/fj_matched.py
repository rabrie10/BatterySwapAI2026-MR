"""Matched near-threshold study: what separates a stable low state from a terminal one?

The structural complaint this answers. The first-passage law drives the crossing
probability to one as the margin goes to zero, and the data contains devices that
sit at a margin of 0.005 V for years without ever crossing -- six of them carry
half of V8's wasted swaps. So among batteries at *the same low margin*, is there
anything in the trajectory that ranks the ones about to die above the ones that
will still be there in three months?

The design is a matched case-control, not a pooled AUC:

* population: rows whose margin is inside ``--band`` (default 0 to 0.10 V);
* cases: an EOL record inside 42 days;
* controls: *observed* to survive at least 42 days -- and a long-survivor
  subgroup observed past 90;
* a case and a control are only ever compared **inside the same scenario and the
  same margin bin**, so absolute voltage cannot solve the problem and neither can
  the calendar, the season or the remaining-observation window.

The metric is that matched concordance. V8's own probability is reported on the
identical pairs as the control: if the margin is matched and V8 lands at 0.5, the
incumbent has nothing left to say about these rows, which is exactly the claim.

    python tools/fj_matched.py --band 0.10 --bin 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.terminality import NAMES as TERMINAL_NAMES  # noqa: E402
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402

HORIZON = 42.0
LONG = 90.0


def matched_pairs(
    scenario: np.ndarray,
    margin_bin: np.ndarray,
    case: np.ndarray,
    control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(case, control) row pairs sharing a scenario and a margin bin."""
    cases: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    key = scenario.astype(np.int64) * 1000 + margin_bin.astype(np.int64)
    for value in np.unique(key):
        rows = np.flatnonzero(key == value)
        a = rows[case[rows]]
        b = rows[control[rows]]
        if a.size == 0 or b.size == 0:
            continue
        ai, bi = np.meshgrid(np.arange(a.size), np.arange(b.size), indexing="ij")
        cases.append(a[ai.ravel()])
        controls.append(b[bi.ravel()])
    if not cases:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(cases), np.concatenate(controls)


def concordance(score: np.ndarray, cases: np.ndarray, controls: np.ndarray,
                weight: np.ndarray | None = None) -> tuple[float, int]:
    a, b = score[cases], score[controls]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() == 0:
        return float("nan"), 0
    wins = (a[good] > b[good]).astype(float) + 0.5 * (a[good] == b[good])
    w = np.ones(good.sum()) if weight is None else weight[good]
    return float((wins * w).sum() / w.sum()), int(good.sum())


def device_weights(battery: np.ndarray, cases: np.ndarray) -> np.ndarray:
    """One unit of weight per distinct case device, spread over its pairs."""
    devices, inverse, counts = np.unique(
        battery[cases], return_inverse=True, return_counts=True
    )
    return 1.0 / counts[inverse]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--terminality", type=Path,
                        default=Path("outputs/fj_terminality.npz"))
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--bin", type=float, default=0.01)
    parser.add_argument("--long", action="store_true",
                        help="controls must be observed past 90 days, not 42")
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_matched.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    margin = frame.column("voltage") - 2.4

    extra = np.load(args.terminality, allow_pickle=False)
    terminal = extra["features"].astype(float)
    signals: dict[str, np.ndarray] = {
        name: terminal[:, index] for index, name in enumerate(TERMINAL_NAMES)
    }
    for index, name in enumerate(FEATURE_NAMES):
        signals.setdefault(name, frame.features[:, index].astype(float))
    signals["p_v8"] = base
    signals["margin"] = margin

    horizon = LONG if args.long else HORIZON
    in_band = (margin >= 0.0) & (margin <= args.band)
    case = in_band & frame.due
    control = (
        in_band
        & ~frame.due
        & (
            (np.isfinite(frame.days_to_eol) & (frame.days_to_eol > horizon))
            | (frame.remaining >= horizon)
        )
    )
    margin_bin = np.floor(margin / args.bin).astype(int)

    cases, controls = matched_pairs(frame.scenario, margin_bin, case, control)
    weight = device_weights(frame.battery, cases)
    print(f"band 0 to {args.band} V, matched in {args.bin} V bins, "
          f"controls survive > {horizon:.0f} d")
    print(f"  cases    {int(case.sum()):5d} rows from "
          f"{np.unique(frame.battery[case]).size:3d} devices")
    print(f"  controls {int(control.sum()):5d} rows from "
          f"{np.unique(frame.battery[control]).size:3d} devices")
    print(f"  matched pairs {cases.size} "
          f"({np.unique(frame.battery[cases]).size} case devices, "
          f"{np.unique(frame.battery[controls]).size} control devices)")
    if cases.size == 0:
        raise SystemExit("no matched pairs")
    print()

    rows = []
    for name, column in signals.items():
        value, count = concordance(column, cases, controls, weight)
        if not np.isfinite(value) or count < 200:
            continue
        rows.append({
            "signal": name,
            "concordance": round(value, 4),
            "edge": round(abs(value - 0.5), 4),
            "direction": "higher" if value > 0.5 else "lower",
            "pairs": count,
            "coverage": round(float(np.isfinite(column[case | control]).mean()), 3),
        })
    rows.sort(key=lambda r: -r["edge"])
    table = pd.DataFrame(rows)
    print("device-weighted matched concordance (0.500 = no information):")
    print(table.head(26).to_string(index=False))
    control_row = table[table.signal == "p_v8"]
    print()
    print("the incumbent on the identical matched pairs:")
    print(control_row.to_string(index=False))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "band": args.band, "bin": args.bin, "control_horizon": horizon,
        "cases": int(case.sum()), "controls": int(control.sum()),
        "case_devices": int(np.unique(frame.battery[case]).size),
        "control_devices": int(np.unique(frame.battery[control]).size),
        "pairs": int(cases.size), "signals": rows,
    }, indent=1))


if __name__ == "__main__":
    main()
