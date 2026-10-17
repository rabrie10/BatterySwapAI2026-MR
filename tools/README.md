# Tool index

Run commands from the repository root. The [reproducibility guide](../docs/REPRODUCIBILITY.md) provides setup and data requirements.

## Supported project entry points

| Tool | Purpose |
|---|---|
| `demo.py` | Synthetic fleet walkthrough; optionally loads the shipped Wiener model |
| `train_wiener.py` | Current model training and building-grouped forecast evaluation |
| `validate_v6.py` | End-to-end evaluation; historical filename, current shipped defaults |
| `check_portfolio.py` | Documentation links, evidence calculations, and saved report assertions |
| `render_results.py` | Regenerate the README chart from the score ledger; needs matplotlib |

## Research and historical tools

Other scripts preserve competition experiments. Examples include `train_v6.py` (hazard model), `train_v7.py` (margin model), `train_aft.py`, and `train_blend.py`, as well as ranking probes, calibration experiments, audits, and tactical planner benchmarks.

These scripts can have historical defaults, model dependencies, or dataset assumptions. Read their module documentation and the matching [experiment notes](../docs/README.md) before use. Their results are not automatically measurements of the current `script.py` configuration.
