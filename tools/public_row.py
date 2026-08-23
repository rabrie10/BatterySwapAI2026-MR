"""Decode a public-leaderboard row into the decision metrics it hides.

The leaderboard publishes cost components, not decisions. Every transfer
question this project has -- did we cut too much volume? is the ranking
finding dues inside the budget? -- is a question about swaps, misses,
precision and recall, and each of those is recoverable in closed form from
two published numbers.

    swaps   = battery_swap / SWAP_PRICE            (0.25 per swap)
    misses  = the m solving late_swap/10 = 27.26*m + m(m-1)/2
    planned = swaps - misses      (a miss is an emergency swap we did not plan)
    caught  = dues - misses       (dues ~ 9.5 per scenario)
    precision = caught / planned ;  recall = caught / dues

The late-swap curve is the queue formula: emergencies collide with each other,
so the m-th one waits longer than the first -- hence the quadratic term. That
convexity is why recall-starving is punished harder than over-swapping (the
handoff's V19 row: -274 early bought +403 late).

All inputs and outputs are PER SCENARIO. Divide leaderboard totals by the
scenario count first.

    python tools/public_row.py --selftest
    python tools/public_row.py --label V19 --swaps 17.8 --late-swap 1037 \
        --early-swap 672 --other 330 --against baseline
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

SWAP_PRICE = 0.25
LATE_UNIT = 10.0
LATE_FIRST = 27.26
DUES_PER_SCENARIO = 9.5


def misses_from_late(late_swap: float) -> float:
    """Invert late(m) = 10*[27.26*m + m(m-1)/2] for m >= 0.

    m^2 + (2*27.26 - 1)*m - 2*late/10 = 0, positive root.
    """
    if late_swap <= 0.0:
        return 0.0
    b = 2.0 * LATE_FIRST - 1.0
    c = -2.0 * late_swap / LATE_UNIT
    return (-b + math.sqrt(b * b - 4.0 * c)) / 2.0


def late_from_misses(misses: float) -> float:
    """Forward direction, so the inversion can be checked against itself."""
    return LATE_UNIT * (LATE_FIRST * misses + misses * (misses - 1.0) / 2.0)


@dataclass(frozen=True)
class Row:
    label: str
    swaps: float
    misses: float
    planned: float
    caught: float
    precision: float
    recall: float
    early_swap: float
    late_swap: float
    other: float
    total: float


def decode(
    label: str,
    *,
    battery_swap: float | None = None,
    swaps: float | None = None,
    late_swap: float,
    early_swap: float = 0.0,
    other: float = 0.0,
    dues: float = DUES_PER_SCENARIO,
) -> Row:
    if swaps is None:
        if battery_swap is None:
            raise ValueError("give either battery_swap or swaps")
        swaps = battery_swap / SWAP_PRICE
    misses = misses_from_late(late_swap)
    planned = swaps - misses
    caught = dues - misses
    return Row(
        label=label,
        swaps=swaps,
        misses=misses,
        planned=planned,
        caught=caught,
        precision=caught / planned if planned > 0 else float("nan"),
        recall=caught / dues if dues > 0 else float("nan"),
        early_swap=early_swap,
        late_swap=late_swap,
        other=other,
        total=swaps * SWAP_PRICE + early_swap + late_swap + other,
    )


def render(rows: list[Row]) -> str:
    head = (
        f"{'row':<12}{'swaps':>7}{'planned':>9}{'misses':>8}{'caught':>8}"
        f"{'prec':>7}{'recall':>8}{'early':>8}{'late':>8}{'other':>8}{'total':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.label:<12}{r.swaps:>7.1f}{r.planned:>9.1f}{r.misses:>8.2f}"
            f"{r.caught:>8.2f}{r.precision:>7.3f}{r.recall:>8.3f}"
            f"{r.early_swap:>8.0f}{r.late_swap:>8.0f}{r.other:>8.0f}{r.total:>8.0f}"
        )
    if len(rows) == 2:
        a, b = rows
        lines += [
            "",
            f"delta {b.label} - {a.label}:",
            f"  swaps   {b.swaps - a.swaps:+.1f}   planned {b.planned - a.planned:+.1f}"
            f"   misses {b.misses - a.misses:+.2f}",
            f"  early   {b.early_swap - a.early_swap:+.0f}   late    "
            f"{b.late_swap - a.late_swap:+.0f}   other  {b.other - a.other:+.0f}"
            f"   total {b.total - a.total:+.0f}",
            "  " + _verdict(a, b),
        ]
    return "\n".join(lines)


def _verdict(a: Row, b: Row) -> str:
    """The section-5 reading rule, stated as the direction to move volume."""
    early_down = b.early_swap < a.early_swap
    late_up = b.late_swap > a.late_swap
    if early_down and late_up:
        return (
            "early down + late up = cut too much volume; the swaps we dropped "
            "were real dues. LOOSEN (raise MAX_PLANNED / relax the multiplier)."
        )
    if not early_down and not late_up:
        return "early up + late flat = over-swapping waste. TIGHTEN."
    if early_down and not late_up:
        return "early down + late down = better ranking at this volume. Hold."
    return "early up + late up = worse on both axes; the ranking regressed."


# The two rows the handoff records (docs/PIHYBRID_HANDOFF.md section 4). If the
# decoder cannot reproduce the misses and recall stated there, it is wrong.
SELFTEST = [
    # label, swaps, early, late, other, expected misses, expected recall
    ("baseline", 21.7, 946.0, 634.0, 410.0, 2.3, 0.76),
    ("V19", 17.8, 672.0, 1037.0, 330.0, 3.6, 0.62),
]


def selftest() -> int:
    ok = True
    rows = []
    for label, swaps, early, late, other, exp_m, exp_r in SELFTEST:
        r = decode(label, swaps=swaps, late_swap=late, early_swap=early, other=other)
        rows.append(r)
        dm, dr = abs(r.misses - exp_m), abs(r.recall - exp_r)
        good = dm < 0.05 and dr < 0.005
        ok &= good
        print(
            f"{label:<10} misses {r.misses:.2f} (doc {exp_m})  "
            f"recall {r.recall:.3f} (doc {exp_r})  {'OK' if good else 'FAIL'}"
        )
    for m in (0.5, 2.27, 3.63, 8.0):
        back = misses_from_late(late_from_misses(m))
        good = abs(back - m) < 1e-9
        ok &= good
        print(f"round-trip m={m}: {back:.12f} {'OK' if good else 'FAIL'}")
    print()
    print(render(rows))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--label", default="row")
    p.add_argument("--battery-swap", type=float, default=None,
                   help="per-scenario battery_swap cost component")
    p.add_argument("--swaps", type=float, default=None,
                   help="per-scenario swap count, if already decoded")
    p.add_argument("--late-swap", type=float, default=None)
    p.add_argument("--early-swap", type=float, default=0.0)
    p.add_argument("--other", type=float, default=0.0,
                   help="capacity + overtime + anything else")
    p.add_argument("--dues", type=float, default=DUES_PER_SCENARIO)
    p.add_argument("--against", default=None,
                   choices=[name for name, *_ in SELFTEST],
                   help="compare with a recorded row and print the volume verdict")
    args = p.parse_args()

    if args.selftest:
        return selftest()
    if args.late_swap is None:
        p.error("--late-swap is required (or use --selftest)")

    rows = []
    if args.against:
        label, swaps, early, late, other, *_ = next(
            r for r in SELFTEST if r[0] == args.against
        )
        rows.append(
            decode(label, swaps=swaps, late_swap=late, early_swap=early,
                   other=other, dues=args.dues)
        )
    rows.append(
        decode(
            args.label,
            battery_swap=args.battery_swap,
            swaps=args.swaps,
            late_swap=args.late_swap,
            early_swap=args.early_swap,
            other=args.other,
            dues=args.dues,
        )
    )
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
