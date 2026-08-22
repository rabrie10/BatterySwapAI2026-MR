"""ROADBLOCK L3: process audit from the validation-run record.

Reads every outputs/val_*.json, extracts the mean, key components, runtime and
file mtime, reconstructs the session's A/B decisions, and classifies each
decision against the measurement-noise model (reroll +-52 on identical
configs; ~100 scenario-overlap floor for design changes).

Output: outputs/roadblock_l3.json + printed table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REROLL = 52.0  # measured no-op rerun drift on the 48-scenario mean
FLOOR = 100.0  # scenario-overlap noise floor per WORKPLAN decision rules


def main() -> None:
    rows = []
    for path in sorted((ROOT / "outputs").glob("val_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        summary = data.get("summary", {})
        mean = summary.get("mean_total_cost")
        if mean is None:
            continue
        runtime = summary.get("runtime", {})
        rows.append(
            {
                "file": path.name,
                "mean": round(float(mean), 1),
                "late": round(float(summary.get("components", {}).get("late_swap", float("nan"))), 1),
                "early": round(float(summary.get("components", {}).get("early_swap", float("nan"))), 1),
                "served": summary.get("decisions", {}).get("served_per_scenario"),
                "sec_per_scen": runtime.get("mean_seconds_per_scenario"),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="minutes"
                ),
            }
        )

    rows.sort(key=lambda r: r["mtime"])

    # Reconstructed decisions of the session (config vs its recorded baseline).
    # (name, arm_file, base_file, call_made, kind)
    decisions = [
        ("candidate-margin -15", "val_margin15.json", "val_ship.json", "kill (noise)", "design"),
        ("calibration clamp<=1", "val_clamp.json", "val_ship.json", "kill (worse)", "design"),
        ("search 240/140", "val_search240.json", "val_ship.json", "keep (sub-noise, structural)", "design"),
        ("reliability isotonic", "val_reliability.json", "val_ship.json", "kill (worse)", "design"),
        ("dwell+iso on v7", "val_dwell.json", "val_ship.json", "kill (worse)", "design"),
        ("cens+dwell+iso", "val_cens_dwell.json", "val_ship.json", "kill (worse)", "design"),
        ("cens + orig calibration", "val_cens_orig.json", "val_ship.json", "keep (real gain)", "design"),
        ("samples0", "val_samples0.json", "val_ship.json", "keep (structural)", "design"),
        ("s0+search240", "val_s0_search240.json", "val_ship.json", "keep (structural)", "design"),
        ("dwell-only retrain", "val_dwellonly.json", "val_ship.json", "kill (worse)", "design"),
        ("v7 rank-cal cap15", "val_v7_rank_cap15.json", "val_v7_cap15.json", "kill (worse)", "design"),
        ("knee floor", "val_v7_knee.json", "val_v7_cap15.json", "kill (noise)", "design"),
        ("gated dwell", "val_v7_gateddwell.json", "val_v7_cap15.json", "kill (noise/worse)", "design"),
        ("cap13 vs cap15", "val_v7_cap13.json", "val_v7_cap15.json", "kill (worse)", "design"),
        ("late-mult 1.4", "val_latemult14.json", "val_ship_final.json", "keep-direction", "knob"),
        ("late-mult 1.8", "val_latemult18.json", "val_ship_final.json", "SHIP (op point)", "knob"),
        ("late-mult 2.2", "val_latemult22.json", "val_ship_final.json", "kill (worse than 1.8)", "knob"),
        ("cap17 @ lm1.8", "val_lm18_cap17.json", "val_latemult18.json", "kill (worse)", "knob"),
        ("vol 1.2", "val_vol12.json", "val_ship_final.json", "kill (no compose)", "knob"),
        ("lm1.4+vol1.2", "val_lm14_vol12.json", "val_latemult14.json", "kill", "knob"),
        ("capacity pass", "val_lm18_cappass.json", "val_latemult18.json", "keep (paired harness said -2.5)", "design"),
        ("lm1.8 rerun (no-op)", "val_lm18_rerun.json", "val_latemult18.json", "measurement", "noise-probe"),
        ("x-banded cap", "val_xband_A.json", "val_latemult18.json", "kill (worse)", "design"),
        ("gate-only", "val_gate_only.json", "val_latemult18.json", "kill (flat: substitution-saturated)", "design"),
        ("gate+demotion composite", "val_stackA.json", "val_latemult18.json", "kill (worse)", "design"),
        ("demotion-only", "val_demotion_only.json", "val_latemult18.json", "kill (worse)", "design"),
        ("v12 invariant retrain", "val_v13_full.json", "val_latemult18.json", "harness-gated", "design"),
        ("v9 full-feature retrain", "val_v9full.json", "val_ship.json", "kill", "design"),
    ]

    by_name = {r["file"]: r for r in rows}
    audited = []
    inside_noise_calls = 0
    for name, arm, base, call, kind in decisions:
        a = by_name.get(arm)
        b = by_name.get(base)
        if a is None or b is None:
            continue
        delta = round(a["mean"] - b["mean"], 1)
        band = REROLL if kind == "knob" else FLOOR
        inside = abs(delta) < band
        # A call is noise-exposed if it asserted a direction while |delta| < band.
        exposed = inside and ("kill (worse)" in call or "keep" in call or "SHIP" in call)
        if exposed:
            inside_noise_calls += 1
        audited.append(
            {
                "decision": name,
                "arm": arm,
                "base": base,
                "delta": delta,
                "band": band,
                "inside_noise": bool(inside),
                "call": call,
                "noise_exposed_call": bool(exposed),
            }
        )

    total_runs = len(rows)
    secs = [r["sec_per_scen"] for r in rows if r["sec_per_scen"]]
    mean_cycle_min = (
        round(sum(secs) / len(secs) * 48 / 60.0, 1) if secs else None
    )
    payload = {
        "noise_model": {"reroll": REROLL, "overlap_floor": FLOOR},
        "validation_runs_recorded": total_runs,
        "mean_validation_cycle_minutes": mean_cycle_min,
        "first_run": rows[0]["mtime"] if rows else None,
        "last_run": rows[-1]["mtime"] if rows else None,
        "decisions_audited": len(audited),
        "decisions_inside_noise_band": sum(1 for a in audited if a["inside_noise"]),
        "noise_exposed_calls": inside_noise_calls,
        "runs": rows,
        "decisions": audited,
    }
    out = ROOT / "outputs" / "roadblock_l3.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"{'file':28s} {'mean':>8s} {'late':>8s} {'early':>7s} {'mtime':>17s}")
    for r in rows:
        print(
            f"{r['file']:28s} {r['mean']:8.1f} {r['late']:8.1f} {r['early']:7.1f} {r['mtime']:>17s}"
        )
    print()
    print(f"{'decision':26s} {'delta':>7s} {'band':>5s} {'in-noise':>8s}  call")
    for a in audited:
        print(
            f"{a['decision']:26s} {a['delta']:7.1f} {a['band']:5.0f} "
            f"{str(a['inside_noise']):>8s}  {a['call']}"
        )
    print(
        f"\nruns={total_runs} cycle~{mean_cycle_min}min "
        f"inside_noise={payload['decisions_inside_noise_band']}/{len(audited)} "
        f"noise_exposed_calls={inside_noise_calls}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
