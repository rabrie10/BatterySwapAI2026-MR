"""Fit the KneeBoost floor table out-of-fold and report the evidence.

Reads the raw out-of-fold diagnostic frame (one row per scenario x alive
battery, raw v7 probability, margin, beta_30, remaining, censor-capped due
label), replicates the fold assignment from the fold artifact, fits one floor
table per fold on the other folds' rows, and a production table on everything.

    python tools/fit_knee.py

Writes outputs/knee_floors.json. Cheap: no model inference, frame only.
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

from bsai.knee import (
    BANDED_BETA_EDGES,
    BETA_EDGES,
    MARGIN_EDGES,
    REMAINING_EDGES,
    KneeBoost,
    KneeBoostBanded,
)


def fold_assignment(frame: pd.DataFrame, folds_path: Path) -> pd.Series:
    bundle = joblib.load(folds_path)
    fold_of_building = {
        building: id(model) for building, model in bundle["by_building"].items()
    }
    # id() values differ between runs; relabel deterministically by first
    # appearance so the report is stable.
    order: dict[int, int] = {}
    labels = []
    for building in sorted(fold_of_building):
        key = fold_of_building[building]
        order.setdefault(key, len(order))
    for b in frame.building:
        labels.append(order.get(fold_of_building.get(b, -1), -1))
    return pd.Series(labels, index=frame.index, name="fold")


def cell_report(boost: KneeBoost, frame: pd.DataFrame, p_col: str = "p") -> list[dict]:
    """Realized vs model per cell, on the rows the floor would govern."""
    out = []
    margin = frame.margin.to_numpy()
    beta = frame.beta30.to_numpy()
    rem = frame.remaining.to_numpy()
    due = frame.due.to_numpy()
    p = frame[p_col].to_numpy()
    for mi, (mlo, mhi) in enumerate(zip(boost.margin_edges[:-1], boost.margin_edges[1:])):
        for bi, (blo, bhi) in enumerate(zip(boost.beta_edges[:-1], boost.beta_edges[1:])):
            cell = (
                (margin >= mlo) & (margin < mhi)
                & np.isfinite(beta) & (beta >= blo) & (beta < bhi)
                & (rem >= boost.remaining_gate)
            )
            n = int(cell.sum())
            out.append(
                {
                    "margin": [mlo, mhi],
                    "beta30": [blo, bhi if bhi < 1e8 else None],
                    "n": n,
                    "events": int(due[cell].sum()),
                    "realized": round(float(due[cell].mean()), 4) if n else None,
                    "mean_p": round(float(p[cell].mean()), 4) if n else None,
                    "median_p": round(float(np.median(p[cell])), 5) if n else None,
                    "floor": round(float(boost.floors[mi][bi]), 4),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/frame_oof_raw_beta.parquet"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/knee_floors.json"))
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--shrink", type=float, default=25.0)
    parser.add_argument("--remaining-gate", type=float, default=30.0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.frame)
    frame["fold"] = fold_assignment(frame, args.folds)
    settings = dict(
        remaining_gate=args.remaining_gate,
        min_rows=args.min_rows,
        min_events=args.min_events,
        shrink=args.shrink,
    )

    # Reproduce the mining numbers this floor is built on, so a regression in
    # the frame shows up here first.
    checks = []
    for mlo, mhi, blo, bhi in [
        (0.05, 0.10, 0.010, 0.014),
        (0.05, 0.10, 0.014, 10.0),
        (0.10, 0.15, 0.010, 10.0),
    ]:
        cell = (
            (frame.margin >= mlo) & (frame.margin < mhi)
            & (frame.beta30 >= blo) & (frame.beta30 < bhi)
        )
        checks.append(
            dict(
                margin=[mlo, mhi], beta30=[blo, bhi],
                n=int(cell.sum()),
                realized=round(float(frame.due[cell].mean()), 4),
                mean_p=round(float(frame.p[cell].mean()), 4),
                median_p=round(float(frame.p[cell].median()), 5),
            )
        )
    print("== mining sanity checks (all remaining) ==")
    for c in checks:
        print(f"  margin {c['margin']} beta {c['beta30']}: n={c['n']} "
              f"realized={c['realized']} mean_p={c['mean_p']} median_p={c['median_p']}")

    per_fold: dict[int, KneeBoost] = {}
    oof_floor = np.zeros(len(frame))
    for fold in sorted(frame.fold.unique()):
        others = frame[frame.fold != fold]
        boost = KneeBoost.fit(
            others.margin.to_numpy(),
            others.beta30.to_numpy(),
            others.due.to_numpy(),
            others.remaining.to_numpy(),
            **settings,
        )
        per_fold[int(fold)] = boost
        mask = (frame.fold == fold).to_numpy()
        oof_floor[mask] = boost.floor_for(
            frame.margin.to_numpy()[mask],
            frame.beta30.to_numpy()[mask],
            frame.remaining.to_numpy()[mask],
        )
        print(f"\n== fold {fold} (fitted on other {len(others)} rows) ==")
        print(boost.describe())

    frame["oof_floor"] = oof_floor
    frame["p_boosted"] = np.maximum(frame.p, oof_floor)

    production = KneeBoost.fit(
        frame.margin.to_numpy(),
        frame.beta30.to_numpy(),
        frame.due.to_numpy(),
        frame.remaining.to_numpy(),
        **settings,
    )
    print("\n== production (fitted on everything) ==")
    print(production.describe())

    # The remaining-banded table: the flat cell rates hide a 0.05-to-0.87
    # gradient along the remaining-observation axis (same axis, same confound
    # warning as bsai/calibrate.py). Fitted with a stricter event gate because
    # the extra axis multiplies the ways to fit noise.
    banded_settings = dict(min_rows=40, min_events=25, shrink=25.0)
    banded_per_fold: dict[int, KneeBoostBanded] = {}
    banded_floor = np.zeros(len(frame))
    for fold in sorted(frame.fold.unique()):
        others = frame[frame.fold != fold]
        banded = KneeBoostBanded.fit(
            others.margin.to_numpy(),
            others.beta30.to_numpy(),
            others.due.to_numpy(),
            others.remaining.to_numpy(),
            **banded_settings,
        )
        banded_per_fold[int(fold)] = banded
        mask = (frame.fold == fold).to_numpy()
        banded_floor[mask] = banded.floor_for(
            frame.margin.to_numpy()[mask],
            frame.beta30.to_numpy()[mask],
            frame.remaining.to_numpy()[mask],
        )
    banded_production = KneeBoostBanded.fit(
        frame.margin.to_numpy(),
        frame.beta30.to_numpy(),
        frame.due.to_numpy(),
        frame.remaining.to_numpy(),
        **banded_settings,
    )
    print("\n== banded production (margin x beta x remaining) ==")
    print(banded_production.describe())
    banded_lift = frame[banded_floor > frame.p]
    print(f"banded OOF floored rows: {len(banded_lift)} ({len(banded_lift)/48:.1f}/scenario), "
          f"dues {int(banded_lift.due.sum())}, realized {banded_lift.due.mean():.3f}")

    lifted = frame[frame.oof_floor > frame.p]
    print(f"\nOOF floored rows: {len(lifted)} ({len(lifted)/48:.1f}/scenario), "
          f"dues among them {int(lifted.due.sum())} "
          f"({int(lifted.due.sum())/48:.2f}/scenario), realized {lifted.due.mean():.3f}, "
          f"mean floor {lifted.oof_floor.mean():.3f}, mean raw p {lifted.p.mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "settings": settings,
                "margin_edges": list(MARGIN_EDGES),
                "beta_edges": list(BETA_EDGES),
                "mining_checks": checks,
                "production": {
                    "floors": [list(r) for r in production.floors],
                    "cells": cell_report(production, frame),
                },
                "per_fold": {
                    str(f): {
                        "floors": [list(r) for r in b.floors],
                        "cell_rows": [list(r) for r in b.cell_rows],
                        "cell_events": [list(r) for r in b.cell_events],
                    }
                    for f, b in per_fold.items()
                },
                "oof": {
                    "floored_rows": int(len(lifted)),
                    "floored_rows_per_scenario": round(len(lifted) / 48, 2),
                    "dues_among_floored": int(lifted.due.sum()),
                    "realized_among_floored": round(float(lifted.due.mean()), 4),
                },
                "banded": {
                    "settings": banded_settings,
                    "remaining_edges": list(REMAINING_EDGES),
                    "beta_edges": list(BANDED_BETA_EDGES),
                    "production": [
                        {
                            "remaining": [lo, hi if hi < 1e8 else None],
                            "floors": [list(r) for r in b.floors],
                            "cell_rows": [list(r) for r in b.cell_rows],
                            "cell_events": [list(r) for r in b.cell_events],
                        }
                        for (lo, hi), b in zip(
                            zip(REMAINING_EDGES[:-1], REMAINING_EDGES[1:]),
                            banded_production.boosts,
                        )
                    ],
                    "per_fold": {
                        str(f): [[list(r) for r in b.floors] for b in banded.boosts]
                        for f, banded in banded_per_fold.items()
                    },
                    "oof": {
                        "floored_rows": int(len(banded_lift)),
                        "dues_among_floored": int(banded_lift.due.sum()),
                        "realized_among_floored": round(float(banded_lift.due.mean()), 4),
                    },
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
