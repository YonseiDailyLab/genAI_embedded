"""Pretrain TinyAE on real or synthetic normal-condition data.

Hyperparameters match on-device training in pipeline.cpp:
  Optimizer : Adam, lr=0.01
  Batch     : 16  (kTrainBatch)
  Loss      : MSE
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import TinyAE
from data import RealSensorDataset, SyntheticNormalDataset, window_to_stats, _load_jsonl_window

import json

BATCH_SIZE     = 16
LR             = 0.001
EPOCHS         = 100
EARLY_STOP_MSE = 0.00005
VAL_FRAC       = 0.1

# This model (6304 params, batch 16×90) is too small for GPU kernel launch overhead.
# CPU is ~1.7x faster than CUDA on Jetson Orin for this workload.
DEVICE = torch.device("cpu")


# ── Fire evaluation ───────────────────────────────────────────────────────────

def _load_fire_tensor(data_dir: Path) -> torch.Tensor | None:
    """Load all anomaly_flag=True samples from labeled JSONL files."""
    samples = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["metadata"].get("anomaly_flag", False):
                    samples.append(window_to_stats(_load_jsonl_window(r)))
    if not samples:
        return None
    return torch.tensor(np.stack(samples))  # [n_fire, 90]


def _eval_separation(model: TinyAE, val_loader: DataLoader,
                     fire_x: torch.Tensor | None, n_val: int) -> dict:
    """Compute normal val MSE and fire detection metrics."""
    model.eval()
    crit = nn.MSELoss(reduction="none")

    with torch.no_grad():
        # Normal val MSE per sample
        n_mse_list = []
        for x in val_loader:
            n_mse_list.append(crit(model(x), x).mean(dim=1))
        n_mse = np.array(torch.cat(n_mse_list).tolist())

        # Tukey IQR thresholds from normal val distribution
        q1, q3 = np.percentile(n_mse, 25), np.percentile(n_mse, 75)
        iqr = q3 - q1
        thr_warn  = q3 + 1.5 * iqr
        thr_anom  = q3 + 3.0 * iqr

        result = {
            "val_mse":   float(n_mse.mean()),
            "thr_warn":  float(thr_warn),
            "thr_anom":  float(thr_anom),
            "fp_warn":   float((n_mse > thr_warn).mean()),   # false positive rate
        }

        if fire_x is not None:
            f_mse = np.array(crit(model(fire_x), fire_x).mean(dim=1).tolist())
            result["fire_mse"]    = float(f_mse.mean())
            result["det_warn"]    = float((f_mse > thr_warn).mean())
            result["det_anom"]    = float((f_mse > thr_anom).mean())
            result["mse_ratio"]   = float(f_mse.mean() / max(n_mse.mean(), 1e-12))

    return result


# ── Dataset builder ───────────────────────────────────────────────────────────

def _build_dataset(data_dir: Path, window_agg: int) -> torch.utils.data.Dataset:
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if jsonl_files:
        print(f"Real data: {len(jsonl_files)} files from {data_dir}")
        ds = RealSensorDataset(jsonl_files, normal_only=True,
                               window_agg=window_agg, device=DEVICE)
        mb = ds.data.element_size() * ds.data.numel() / 1e6
        print(f"  → {len(ds)} normal windows  ({mb:.1f} MB on {DEVICE})")
    else:
        print(f"No JSONL in {data_dir} — using synthetic data")
        ds = SyntheticNormalDataset(n_windows=10_000, seed=42, device=DEVICE)
    return ds


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> TinyAE:
    print(f"device : {DEVICE}")

    data_dir = Path(args.data_dir)
    dataset  = _build_dataset(data_dir, args.window_agg)

    n_val   = max(1, int(VAL_FRAC * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=512)

    # Load fire samples for per-epoch separation tracking.
    fire_x = _load_fire_tensor(data_dir)
    if fire_x is not None:
        print(f"  → {len(fire_x)} fire samples loaded for eval")
    else:
        print("  → no fire samples found (separation metrics unavailable)")

    model = TinyAE().to(DEVICE)
    model.init_glorot()
    print(f"TinyAE weights : {model.weight_count}  (90-32-6-32-90)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    out_path  = Path(args.output)
    best_val  = float("inf")

    if args.resume and out_path.exists():
        model.load_state_dict(
            torch.load(out_path, map_location=DEVICE, weights_only=True))
        print(f"resumed from {out_path}")

    # Header
    if fire_x is not None:
        print(f"\n{'ep':>5}  {'train':>9}  {'val':>9}  {'thr_anom':>9}  "
              f"{'fire_mse':>9}  {'det%':>6}  {'FP%':>6}  {'ratio':>6}")
        print("-" * 73)
    else:
        print(f"\n{'ep':>5}  {'train':>9}  {'val':>9}  {'thr_anom':>9}  {'FP%':>6}")
        print("-" * 46)

    pbar = tqdm(range(1, args.epochs + 1), unit="ep")
    for epoch in pbar:
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x in train_loader:
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # ── validate ──────────────────────────────────────────────────────
        ev = _eval_separation(model, val_loader, fire_x, n_val)

        if fire_x is not None:
            pbar.write(f"{epoch:5d}  {train_loss:9.6f}  {ev['val_mse']:9.6f}  "
                       f"{ev['thr_anom']:9.6f}  {ev['fire_mse']:9.6f}  "
                       f"{ev['det_anom']*100:5.1f}%  {ev['fp_warn']*100:5.2f}%  "
                       f"{ev['mse_ratio']:6.1f}x")
        else:
            pbar.write(f"{epoch:5d}  {train_loss:9.6f}  {ev['val_mse']:9.6f}  "
                       f"{ev['thr_anom']:9.6f}  {ev['fp_warn']*100:5.2f}%")

        if ev["val_mse"] < best_val:
            best_val = ev["val_mse"]
            torch.save(model.state_dict(), out_path)

        if ev["val_mse"] < args.early_stop:
            pbar.write(f"\nearly stop  epoch={epoch}  val={ev['val_mse']:.6f}")
            break

    print(f"\nbest val MSE : {best_val:.6f}  →  {out_path}")
    model.load_state_dict(
        torch.load(out_path, map_location=DEVICE, weights_only=True))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="../data/labeled")
    parser.add_argument("--window-agg",  type=int,   default=1)
    parser.add_argument("--output",                  default="tinyae_pretrained.pt")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--early-stop",  type=float, default=EARLY_STOP_MSE)
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    train(parser.parse_args())
