"""Check local links and quantified claims in the current portfolio documents."""

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = ["README.md", "docs/README.md", "docs/RESULTS.md", "docs/CASE_STUDY.md",
                "docs/ARCHITECTURE.md", "docs/REPRODUCIBILITY.md", "docs/DEMO.md",
                "models/README.md", "notebooks/README.md", "tools/README.md", "src/README.md"]


def main():
    errors = []
    for name in CURRENT_DOCS:
        path = ROOT / name
        content = path.read_text(encoding="utf-8-sig")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            if not (path.parent / unquote(parsed.path)).resolve().exists():
                errors.append(f"{name}: missing link target {target}")

    ledger = json.loads((ROOT / "docs/results.json").read_text())
    report = json.loads((ROOT / ledger["training_report"]).read_text())
    first, last = ledger["submissions"][0]["cost"], ledger["submissions"][-1]["cost"]
    reduction = 100 * (1 - last / first)
    local_reduction = 100 * (1 - 2293.2 / 3324.7)
    metrics = report["metrics"]["scale_1.0"]
    latest = ledger["latest_leaderboard"]
    component_sum = sum(latest["components"].values())
    if round(component_sum, 2) != latest["total_time"]:
        errors.append("Leaderboard components do not match total time after rounding")
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    claims = [f"{reduction:.1f}%", f"{local_reduction:.1f}%", f"{metrics['auc']:.4f}",
              f"{metrics['precision_at_50']:.0%}", f"{latest['total_time']:.2f}",
              f"{latest['rank']}th place", f"{report['n_cutoffs']:,}"]
    for claim in claims:
        if claim not in readme:
            errors.append(f"README is missing the ledger/report-derived claim: {claim}")
    svg = ROOT / "docs/assets/results.svg"
    ET.parse(svg)
    if f"{reduction:.1f}%" not in svg.read_text(encoding="utf-8"):
        errors.append("Results figure is stale; run tools/render_results.py")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Checked {len(CURRENT_DOCS)} documents, result arithmetic, saved metrics, and SVG.")
    print(f"Leaderboard reduction: {reduction:.2f}%; local reduction: {local_reduction:.2f}%")
    print(f"Leaderboard component sum: {component_sum:.4f}")


if __name__ == "__main__":
    main()
