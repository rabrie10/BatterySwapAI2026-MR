"""Validate a generated ``submission.csv`` the way the evaluator will read it.

Run this in a clean checkout after ``script.py``, before anything is submitted.
It checks the plan itself, not the score: schema, coverage, duplicates, dates,
the deferral rule and the numeric health of the whole file. It also re-loads the
model artifact through the submission's own loader, because a Git-LFS pointer
unpickles as garbage rather than failing loudly.

    python tools/fj_check_submission.py --submission submission.csv --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset  # noqa: E402


def fail(message: str) -> None:
    print(f"  FAIL  {message}")
    fail.count += 1


fail.count = 0


def ok(message: str) -> None:
    print(f"  ok    {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=Path, default=None,
                        help="artifact to load through the submission's own loader")
    args = parser.parse_args()

    print(f"== artifact ==")
    import script

    path = args.model or (REPO_ROOT / script.DEFAULT_MODEL_PATH)
    if not path.exists():
        fail(f"{path} does not exist")
    else:
        head = path.read_bytes()[:64]
        if b"git-lfs" in head or b"oid sha256" in head:
            fail(f"{path} is a Git-LFS pointer, not a model")
        else:
            ok(f"{path} is a real file, {path.stat().st_size / 1e6:.2f} MB")
        import joblib

        model = joblib.load(path)
        ok(script._describe_model(model, path))
        if getattr(model, "model_version", None) != script.INCUMBENT_MODEL_VERSION:
            fail(f"loaded {getattr(model, 'model_version', None)}, "
                 f"not the incumbent {script.INCUMBENT_MODEL_VERSION}")

    print(f"== submission ==")
    if not args.submission.exists():
        fail(f"{args.submission} does not exist")
        raise SystemExit(1)
    frame = pd.read_csv(args.submission)
    ok(f"{len(frame)} rows, columns {list(frame.columns)}")
    for column in ("day", "battery", "scenario", "split"):
        if column not in frame.columns:
            fail(f"missing column {column!r}")
    if frame.isna().any().any():
        fail("the file contains NaN")
    else:
        ok("no NaN anywhere")
    days = pd.to_datetime(frame["day"], errors="coerce")
    if days.isna().any():
        fail("some day values do not parse as dates")
    else:
        ok(f"all days parse; {days.min().date()} .. {days.max().date()}")

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset / args.split)
    expected = 0
    problems = 0
    for scenario, locs, cut, _ in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        name = scenario["name"]
        rows = frame[frame["scenario"] == name]
        active = set(locs["battery"].astype(str))
        expected += len(active)
        planned = rows["battery"].astype(str)
        if len(planned) != len(set(planned)):
            fail(f"{name}: duplicated batteries")
            problems += 1
        if set(planned) != active:
            fail(f"{name}: {len(active - set(planned))} missing, "
                 f"{len(set(planned) - active)} unexpected")
            problems += 1
        start = pd.Timestamp(scenario["start_time"]).normalize()
        window = float(scenario["settings"].planning_window_days)
        offsets = (pd.to_datetime(rows["day"]).dt.normalize() - start) / pd.Timedelta(days=1)
        if (offsets < 0).any():
            fail(f"{name}: {int((offsets < 0).sum())} rows before the scenario start")
            problems += 1
        if problems > 6:
            fail("too many scenario problems; stopping the per-scenario scan")
            break
    if problems == 0:
        ok(f"{len(scenarios)} scenarios: every active battery appears exactly once, "
           f"no duplicates, no day before the start ({expected} rows expected)")

    print()
    if fail.count:
        print(f"{fail.count} FAILURES")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
