"""Regenerate the README figure: python tools/render_results.py (matplotlib)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "docs/results.json").read_text())
    rows = data["submissions"]
    reduction = 100 * (1 - rows[-1]["cost"] / rows[0]["cost"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#101c30")
    ax.set_facecolor("#101c30")
    fig.text(.05, .91, f"Rank 5 | {reduction:.1f}% lower leaderboard cost", color="white",
             size=22, weight="bold")
    fig.text(.05, .84, "Initial public score to latest team-reported leaderboard result",
             color="#c1cede", size=11)
    colors = ["#6f8099", "#6f8099", "#5e9df5", "#5e9df5", "#44d6ae"]
    ax.barh([r["label"] for r in rows], [r["cost"] for r in rows], color=colors, height=.55)
    ax.invert_yaxis()
    for i, row in enumerate(rows):
        ax.text(row["cost"] + 65, i, f'{row["cost"]:,.2f}', va="center", color="white", size=12)
    ax.set_xlim(0, 4950)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.tick_params(axis="both", colors="#c1cede", length=0, labelsize=10, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#304058", linewidth=.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.05, .06, "Team-reported scores • Lower is better • Latest submission: 23 August 2026",
             color="#c1cede", size=10)
    fig.subplots_adjust(left=.25, right=.96, bottom=.19, top=.75)
    target = ROOT / "docs/assets/results.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, metadata={"Date": None, "Description": data["metric"]})
    target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())
                      + "\n", encoding="utf-8")
    preview = ROOT / "outputs/demo/results-preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
