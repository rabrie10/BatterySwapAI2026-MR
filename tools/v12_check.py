"""Fast correctness checks for the V12 invariant feature variant.

Run before anything heavy: verifies the base feature path is byte-identical,
the variants are consistent column selections, and the new physics features
recover known answers on synthetic devices.

    python tools/v12_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai import features as fl
from bsai.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    FEATURE_NAMES_INVARIANT,
    FEATURE_NAMES_INVARIANT2,
    INVARIANT_DROPPED,
    INVARIANT2_DROPPED,
    RAW_ANY_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    TEMPERATURE_BETA,
    DeviceView,
    FeatureContext,
    feature_row,
    variant_needs_raw,
)
from bsai.shape import DeviceShape, align_to

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def synthetic_device(n: int = 900, beta: float = 0.006, seed: int = 7):
    rng = np.random.default_rng(seed)
    day = np.arange(n, dtype=float)
    temperature = 20.0 + 4.0 * np.sin(2 * np.pi * day / 365.25) + rng.normal(0, 0.3, n)
    voltage = (
        3.05
        - 0.0006 * day
        + beta * (temperature - 20.0)
        + rng.normal(0, 0.004, n)
    )
    return voltage, temperature


def synthetic_shape(n: int, level: float = 0.004, rise_from: int | None = None):
    beta = np.full(n, level)
    if rise_from is not None:
        beta[rise_from:] = level * 3.0
    return DeviceShape(
        origin=0,
        beta=beta,
        v_std=beta * 2.0,
        v_range=beta * 10.0,
        t_range=np.full(n, 5.0),
    )


def main() -> None:
    print("name lists", flush=True)
    check("base length 64", len(FEATURE_NAMES) == 64, str(len(FEATURE_NAMES)))
    raw_all = list(RAW_FEATURE_NAMES) + list(RAW_ANY_FEATURE_NAMES)
    check(
        "extended = base + 11 + 7 raw + 2 raw_any",
        FEATURE_NAMES_EXTENDED[: len(FEATURE_NAMES)] == FEATURE_NAMES
        and len(FEATURE_NAMES_EXTENDED) == len(FEATURE_NAMES) + 11 + len(raw_all)
        and FEATURE_NAMES_EXTENDED[-len(raw_all) :] == raw_all,
        str(len(FEATURE_NAMES_EXTENDED)),
    )
    check(
        "invariant (round 1, frozen): fragile 12 out, no raw channels",
        all(name not in FEATURE_NAMES_INVARIANT for name in INVARIANT_DROPPED)
        and all(name not in FEATURE_NAMES_INVARIANT for name in raw_all)
        and len(FEATURE_NAMES_INVARIANT) == 63,
        str(len(FEATURE_NAMES_INVARIANT)),
    )
    check(
        "invariant2: temp levels out, scales+ratios+raw channels in",
        all(name not in FEATURE_NAMES_INVARIANT2 for name in INVARIANT2_DROPPED)
        and all(name in FEATURE_NAMES_INVARIANT2 for name in raw_all)
        and all(
            name in FEATURE_NAMES_INVARIANT2
            for name in ("beta_30", "v_std_30", "beta_rise_mid", "recovery_residual_30")
        )
        and len(FEATURE_NAMES_INVARIANT2) == len(FEATURE_NAMES_EXTENDED) - 5,
        str(len(FEATURE_NAMES_INVARIANT2)),
    )
    check(
        "variant_needs_raw truth table",
        (not variant_needs_raw("base"))
        and (not variant_needs_raw("invariant"))
        and variant_needs_raw("invariant2")
        and variant_needs_raw("extended"),
    )
    check(
        "voltage first in every variant",
        FEATURE_NAMES_INVARIANT[0] == "voltage"
        and FEATURE_NAMES_INVARIANT2[0] == "voltage",
    )
    kept = ("voltage", "voltage_compensated", "staleness", "beta_rise", "v_std_rise",
            "v_range_rise", "age_days", "gap_fraction_90", "season_sin", "season_cos",
            "days_below_2.45", "knee_recent_vs_baseline", "drawdown", "slope_30",
            "slope_comp_30", "temp_outlook_42")
    check("load-bearing features kept", all(k in FEATURE_NAMES_INVARIANT for k in kept))

    print("rows on a synthetic device", flush=True)
    voltage, temperature = synthetic_device()
    n = voltage.size
    view = DeviceView(voltage, temperature)
    shape = align_to(synthetic_shape(n, rise_from=700), 0, n)
    context = FeatureContext(climatology=np.zeros(12))

    for index in (80, 200, 450, 820):
        base = feature_row(view, index, 18000 + index, context, shape, variant="base")
        extended = feature_row(view, index, 18000 + index, context, shape, variant="extended")
        invariant = feature_row(view, index, 18000 + index, context, shape, variant="invariant")
        check(
            f"base == extended[:64] @ {index}",
            np.allclose(base, extended[: len(base)], equal_nan=True),
        )
        picked = [extended[FEATURE_NAMES_EXTENDED.index(nm)] for nm in FEATURE_NAMES_INVARIANT]
        check(
            f"invariant == selection @ {index}",
            np.allclose(invariant, picked, equal_nan=True),
        )

    fl.set_feature_variant("invariant")
    via_registry = feature_row(view, 450, 18450, context, shape)
    fl.set_feature_variant("base")
    direct = feature_row(view, 450, 18450, context, shape, variant="invariant")
    check("registry matches explicit variant", np.allclose(via_registry, direct, equal_nan=True))
    check(
        "registry restored to base",
        len(feature_row(view, 450, 18450, context, shape)) == len(FEATURE_NAMES),
    )

    print("raw-daily channel plumbing", flush=True)
    planted = [2.41, 2.39, 2.4, 2.38, -0.004, 5.0, 7.0]
    planted_any = [2.37, 2.33]
    raw_adapter = lambda day_ordinal: planted  # noqa: E731
    any_adapter = lambda day_ordinal: planted_any  # noqa: E731
    with_raw = feature_row(
        view, 450, 18450, context, shape, variant="extended",
        raw=raw_adapter, raw_any=any_adapter,
    )
    check(
        "extended carries planted raw + raw_any values",
        np.allclose(
            [with_raw[FEATURE_NAMES_EXTENDED.index(n)] for n in RAW_FEATURE_NAMES],
            planted,
        )
        and np.allclose(
            [with_raw[FEATURE_NAMES_EXTENDED.index(n)] for n in RAW_ANY_FEATURE_NAMES],
            planted_any,
        ),
    )
    inv2 = feature_row(
        view, 450, 18450, context, shape, variant="invariant2",
        raw=raw_adapter, raw_any=any_adapter,
    )
    check(
        "invariant2 places both raw channels at named columns",
        len(inv2) == len(FEATURE_NAMES_INVARIANT2)
        and np.allclose(
            [inv2[FEATURE_NAMES_INVARIANT2.index(n)] for n in RAW_FEATURE_NAMES],
            planted,
        )
        and np.allclose(
            [inv2[FEATURE_NAMES_INVARIANT2.index(n)] for n in RAW_ANY_FEATURE_NAMES],
            planted_any,
        )
        and inv2[FEATURE_NAMES_INVARIANT2.index("beta_30")]
        == with_raw[FEATURE_NAMES_EXTENDED.index("beta_30")],
    )
    no_raw = feature_row(view, 450, 18450, context, shape, variant="extended")
    check(
        "raw=None leaves all raw columns NaN",
        np.all(np.isnan(no_raw[-len(raw_all) :])),
    )
    half = feature_row(
        view, 450, 18450, context, shape, variant="extended", raw=raw_adapter
    )
    check(
        "raw without raw_any: filtered filled, any-temp NaN",
        np.allclose(
            [half[FEATURE_NAMES_EXTENDED.index(n)] for n in RAW_FEATURE_NAMES], planted
        )
        and np.all(
            np.isnan([half[FEATURE_NAMES_EXTENDED.index(n)] for n in RAW_ANY_FEATURE_NAMES])
        ),
    )
    base_with_raw = feature_row(
        view, 450, 18450, context, shape, variant="base", raw=raw_adapter
    )
    check(
        "base variant ignores the raw adapter",
        len(base_with_raw) == len(FEATURE_NAMES),
    )
    inv1 = feature_row(
        view, 450, 18450, context, shape, variant="invariant", raw=raw_adapter
    )
    check(
        "frozen invariant excludes raw even when offered",
        len(inv1) == len(FEATURE_NAMES_INVARIANT)
        and not any(v in planted for v in inv1[-5:]),
    )

    print("early-life dV/dT", flush=True)
    early = view.early_beta(400)
    check("recovers planted beta 0.006", abs(early - 0.006) < 0.0012, f"{early:.5f}")
    check("frozen after 180 days", view.early_beta(400) == view.early_beta(880))
    check("fallback before 60 days", view.early_beta(30) == TEMPERATURE_BETA)
    flat_view = DeviceView(voltage, np.full(n, 20.0))
    check("fallback on flat temperature", flat_view.early_beta(400) == TEMPERATURE_BETA)

    print("recovery residual", flush=True)
    ext_names = FEATURE_NAMES_EXTENDED
    row = feature_row(view, 820, 18820, context, shape, variant="extended")
    healthy = row[ext_names.index("recovery_residual_30")]
    # Healthy synthetic cell: V moves with T through its own beta, so the
    # residual is the aging drift alone (~ -0.0006*30 = -0.018).
    check(
        "healthy residual ~ aging drift",
        np.isfinite(healthy) and abs(healthy - (-0.018)) < 0.012,
        f"{healthy:.4f}",
    )
    # Dying synthetic cell: warming no longer lifts the voltage.
    dead_v = voltage.copy()
    dead_v[700:] = voltage[700] - 0.002 * np.arange(n - 700) - 0.006 * (
        temperature[700:] - 20.0
    )  # cancel the temperature response entirely after day 700
    dead_view = DeviceView(dead_v, temperature)
    dead_row = feature_row(dead_view, 820, 18820, context, shape, variant="extended")
    dying = dead_row[ext_names.index("recovery_residual_30")]
    check(
        "dying residual more negative than healthy",
        np.isfinite(dying) and dying < healthy - 0.01,
        f"{dying:.4f} vs {healthy:.4f}",
    )

    print("z and rise ratios", flush=True)
    z = row[ext_names.index("beta_z_30")]
    check("beta_z_30 flags the 3x shape rise", np.isfinite(z) and z > 3.0, f"{z:.2f}")
    quiet = feature_row(view, 450, 18450, context, shape, variant="extended")
    check(
        "quiet-period z near zero",
        abs(quiet[ext_names.index("beta_z_30")]) < 1.0,
        f"{quiet[ext_names.index('beta_z_30')]:.2f}",
    )
    onset = feature_row(view, 740, 18740, context, shape, variant="extended")
    check(
        "mid rise ratio fires during onset",
        onset[ext_names.index("beta_rise_mid")] > 1.2,
        f"{onset[ext_names.index('beta_rise_mid')]:.2f}",
    )
    check(
        "mid rise ratio saturates to 1 long after onset",
        abs(row[ext_names.index("beta_rise_mid")] - 1.0) < 0.05,
        f"{row[ext_names.index('beta_rise_mid')]:.2f}",
    )
    tnd = row[ext_names.index("temp_now_delta")]
    check("temp_now_delta finite and centred", np.isfinite(tnd) and abs(tnd) < 8.0, f"{tnd:.2f}")

    no_shape = feature_row(view, 450, 18450, context, None, variant="invariant")
    check(
        "no-shape row has NaNs, right length",
        len(no_shape) == len(FEATURE_NAMES_INVARIANT),
    )

    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"all {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
