"""Train the sequence quantile head: 5 folds by building + one production fit.

Same statistical framing and gate as the incumbent censored GBDT
(``tools/train_wiener.py``): censor-aware margin drops over the FIT_HORIZONS
windows, fitted fold-by-building, judged by out-of-fold PR-AUC on the 42-day
decision over the stride-4 cutoff frame. The incumbent's number to beat is
0.4706 (models/v8_cens.joblib training report).

The trunk runs once per cutoff and the quantile head once per (cutoff,
horizon), so the full ~550k-window population trains at CNN-trunk cost of the
~88k cutoffs -- no subsampling needed.

    python tools/seq_pack.py                # once
    python tools/train_seq.py               # 5 folds + production + bundle
    python tools/train_seq.py --benchmark   # timing probe, no artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from bsai.features import FEATURE_NAMES
from bsai.seq_head import (
    SEQ_FIT_HORIZONS,
    SeqModel,
    SeqQuantileNet,
    gather_windows,
    horizon_scalars,
    input_from_windows,
    pinball_loss,
    probability_at,
)

BETA30_COL = FEATURE_NAMES.index("beta_30")

DEFAULT_WORK = Path(
    os.environ.get(
        "SEQ_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\seq",
    )
)

INCUMBENT_PR_AUC = 0.4706


def report(probability: np.ndarray, truth: np.ndarray) -> dict:
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "base_rate": round(float(truth.mean()), 5),
        "auc": round(float(roc_auc_score(truth, probability)), 4),
        "pr_auc": round(float(average_precision_score(truth, probability)), 4),
        "predicted_over_actual": round(float(probability.sum() / max(truth.sum(), 1)), 3),
    }
    order = np.argsort(-probability)
    for k in (50, 100, 200, 500, 1000):
        if k <= truth.size:
            out[f"precision_at_{k}"] = round(float(truth[order[:k]].mean()), 4)
    return out


def train_net(
    rows: np.ndarray,
    train: dict,
    ptr: np.ndarray,
    w_h: np.ndarray,
    w_t: np.ndarray,
    *,
    epochs: int,
    batch: int,
    lr: float,
    width: int,
    seed: int,
    max_steps: int | None = None,
    log_every: int = 40,
) -> SeqQuantileNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SeqQuantileNet(width=width)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    horizons = np.asarray(SEQ_FIT_HORIZONS, dtype=np.float64)

    bank = train["bank_data"]
    offsets = train["bank_offsets"]
    dev = train["frame_device_index"]
    cutoffs = train["frame"].cutoff
    margins = train["frame_margin"].astype(np.float64)

    counts_all = ptr[1:] - ptr[:-1]
    step = 0
    started = time.time()
    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = lr * (0.3 ** epoch)
        perm = rng.permutation(rows)
        net.train()
        running = 0.0
        for lo in range(0, perm.size, batch):
            rows_b = perm[lo : lo + batch]
            counts = counts_all[rows_b]
            keep = counts > 0
            rows_b = rows_b[keep]
            counts = counts[keep]
            if rows_b.size == 0:
                continue
            windows = gather_windows(bank, offsets[dev[rows_b]], cutoffs[rows_b])
            x = torch.from_numpy(input_from_windows(windows))
            starts = ptr[rows_b]
            total = int(counts.sum())
            base = np.repeat(starts, counts)
            offset_in = np.arange(total) - np.repeat(
                np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
            )
            flat = base + offset_in
            pair_local = np.repeat(np.arange(rows_b.size), counts)

            scalars = horizon_scalars(
                margins[rows_b][pair_local], horizons[w_h[flat]]
            )
            target = torch.from_numpy(w_t[flat].astype(np.float32))

            optimizer.zero_grad()
            z = net.encode(x)
            index = torch.from_numpy(pair_local.astype(np.int64))
            predicted = net.head(z[index], torch.from_numpy(scalars))
            loss = pinball_loss(predicted, target)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            step += 1
            if step % log_every == 0:
                print(
                    f"    step {step:4d} epoch {epoch} loss {running/log_every:.4f} "
                    f"({(time.time()-started)/step:.2f}s/step)",
                    flush=True,
                )
                running = 0.0
            if max_steps is not None and step >= max_steps:
                print(f"    stop at {step} steps ({(time.time()-started)/step:.2f}s/step)")
                return net
    return net


def oof_probability(net: SeqQuantileNet, train: dict, rows: np.ndarray) -> np.ndarray:
    windows = gather_windows(
        train["bank_data"],
        train["bank_offsets"][train["frame_device_index"][rows]],
        train["frame"].cutoff[rows],
    )
    return probability_at(
        net,
        windows,
        train["frame_margin"][rows].astype(np.float64),
        train["decision_horizon"][rows].astype(np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_WORK / "seq_pack.joblib")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--benchmark", action="store_true", help="time 30 steps and exit")
    parser.add_argument("--out-folds", type=Path, default=REPO_ROOT / "outputs/seq_folds.joblib")
    parser.add_argument("--out-model", type=Path, default=REPO_ROOT / "outputs/seq_production.joblib")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/seq_oof_report.json")
    parser.add_argument("--oof-dump", type=Path, default=DEFAULT_WORK / "seq_oof.npz")
    args = parser.parse_args()

    torch.set_num_threads(3)
    started = time.time()
    print(f"loading {args.pack}...", flush=True)
    pack = joblib.load(args.pack)
    train, deploy = pack["train"], pack["deploy"]
    frame = train["frame"]

    # CSR over windows: rows sorted, spans per cutoff row.
    order = np.argsort(train["window_row"], kind="stable")
    w_row = train["window_row"][order]
    w_h = train["window_horizon"][order]
    w_t = train["window_target"][order]
    ptr = np.searchsorted(w_row, np.arange(len(frame) + 1)).astype(np.int64)
    print(
        f"{len(frame)} cutoffs, {w_row.size} windows, "
        f"{int(train['truth'].sum())} due rows", flush=True,
    )

    net_probe = SeqQuantileNet(width=args.width)
    n_params = net_probe.parameter_count()
    print(f"model: {n_params} parameters (limit 150k)", flush=True)
    assert n_params <= 150_000

    if args.benchmark:
        rows = np.flatnonzero(ptr[1:] > ptr[:-1])
        train_net(
            rows, train, ptr, w_h, w_t,
            epochs=1, batch=args.batch, lr=args.lr, width=args.width,
            seed=args.seed, max_steps=30, log_every=10,
        )
        return

    splitter = GroupKFold(n_splits=args.folds)
    groups = frame.building[w_row]  # building per window, split like train_wiener
    oof = np.zeros(len(frame), dtype=np.float64)
    fold_nets: list[SeqQuantileNet] = []
    fold_heldout: list[list[str]] = []
    building_fold: dict[str, int] = {}

    for fold, (train_windows, _) in enumerate(
        splitter.split(w_t, w_t, groups)
    ):
        held = sorted(set(np.unique(groups)) - set(np.unique(groups[train_windows])))
        train_rows = np.flatnonzero(
            ~np.isin(frame.building, held) & (ptr[1:] > ptr[:-1])
        )
        print(
            f"fold {fold}: {train_rows.size} cutoffs, held {held} "
            f"({time.time()-started:.0f}s)", flush=True,
        )
        net = train_net(
            train_rows, train, ptr, w_h, w_t,
            epochs=args.epochs, batch=args.batch, lr=args.lr,
            width=args.width, seed=args.seed + fold,
        )
        held_rows = np.flatnonzero(np.isin(frame.building, held))
        oof[held_rows] = oof_probability(net, train, held_rows)
        fold_nets.append(net)
        fold_heldout.append(list(held))
        for building in held:
            building_fold[str(building)] = fold
        print(f"fold {fold} done ({time.time()-started:.0f}s)", flush=True)

    truth = train["truth"].astype(np.int8)
    metrics = {"oof": report(oof, truth)}
    print(json.dumps(metrics, indent=2), flush=True)
    gate_a = metrics["oof"]["pr_auc"] > INCUMBENT_PR_AUC
    print(
        f"GATE (a): OOF PR-AUC {metrics['oof']['pr_auc']} vs incumbent "
        f"{INCUMBENT_PR_AUC} -> {'PASS' if gate_a else 'FAIL'}", flush=True,
    )

    print("production fit on all buildings...", flush=True)
    all_rows = np.flatnonzero(ptr[1:] > ptr[:-1])
    production_net = train_net(
        all_rows, train, ptr, w_h, w_t,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        width=args.width, seed=args.seed + 99,
    )

    # ---- bundle for fit_calibration / validate_v6 --------------------------
    key_index = {key: i for i, key in enumerate(deploy["keys"])}
    shared_windows = deploy["windows"]
    shared_margins = deploy["margin"].astype(np.float32)

    def make_shim(net: SeqQuantileNet) -> SeqModel:
        net.eval()
        return SeqModel(
            net=net,
            windows=shared_windows,
            margins=shared_margins,
            key_index=key_index,
            climatology=train["climatology"],
        )

    shims = [make_shim(net) for net in fold_nets]
    by_building: dict[str, SeqModel] = {}
    for building in sorted(set(map(str, np.unique(frame.building)))):
        by_building[building] = shims[building_fold[building]]
    # Buildings absent from the frame (none expected) fall back to fold 0.
    for building in sorted(set(map(str, train["building_sizes"]))):
        by_building.setdefault(building, shims[0])

    args.out_folds.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"by_building": by_building, "climatology": train["climatology"]},
        args.out_folds,
    )
    joblib.dump(make_shim(production_net), args.out_model)
    print(f"wrote {args.out_folds} and {args.out_model}", flush=True)

    np.savez_compressed(
        args.oof_dump,
        oof=oof,
        truth=truth,
        margin=train["frame_margin"],
        beta30=frame.features[:, BETA30_COL],
        building=frame.building,
        cutoff=frame.cutoff,
        decision_horizon=train["decision_horizon"],
        remaining=(frame.observation_end - frame.cutoff).astype(np.float32),
    )
    print(f"wrote {args.oof_dump}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_version": "bsai-seq/v1",
                "n_parameters": n_params,
                "width": args.width,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "fit_horizons": list(SEQ_FIT_HORIZONS),
                "n_cutoffs": int(len(frame)),
                "n_windows": int(w_row.size),
                "fold_heldout": fold_heldout,
                "metrics": metrics,
                "gate_a_pass": bool(gate_a),
                "incumbent_pr_auc": INCUMBENT_PR_AUC,
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {args.report} in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
