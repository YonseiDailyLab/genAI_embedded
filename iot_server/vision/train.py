"""
비전 이상치 탐지 모델 학습.

데이터: data/data/normal/*.jpg  (정상 ~3960장)
        data/data/fire/*.jpg    (화재   ~40장)

모델: Fire2DCNN (vision/model.py)
      - 경량 2D CNN -> 화재 이진 분류
      - 정적 JPEG에 최적화, Jetson 메모리 제약 고려

클래스 불균형 처리:
  - WeightedRandomSampler (오버샘플링)
  - pos_weight 가중 BCEWithLogitsLoss

Usage:
  cd iot_server && python -m vision.train
  cd iot_server && python -m vision.train --epochs 100 --lr 3e-4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vision.model import Fire2DCNN

# -- 경로 ----------------------------------------------------------------------

DATA_ROOT  = Path(__file__).parent.parent.parent / "data" / "data"
NORMAL_DIR = DATA_ROOT / "normal"
FIRE_DIR   = DATA_ROOT / "fire"
OUT_DIR    = Path(__file__).parent / "checkpoints"

# -- 기본값 --------------------------------------------------------------------

DEFAULTS = {
    "epochs":     100,
    "lr":         3e-4,
    "batch_size": 32,
    "img_size":   112,
    "base_ch":    32,
    "dropout_p":  0.4,
    "val_frac":   0.2,
    "seed":       42,
}


# -- Augmentation (cv2 기반, NumPy 2.x 호환) ----------------------------------

def _aug(img_bgr: np.ndarray, img_size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    # Random resized crop
    scale = float(rng.uniform(0.65, 1.0))
    ch, cw = int(h * scale), int(w * scale)
    y0 = int(rng.integers(0, max(h - ch, 0) + 1))
    x0 = int(rng.integers(0, max(w - cw, 0) + 1))
    img_bgr = img_bgr[y0:y0+ch, x0:x0+cw]

    img_bgr = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if rng.random() > 0.5:
        img_bgr = cv2.flip(img_bgr, 1)   # horizontal flip
    if rng.random() > 0.85:
        img_bgr = cv2.flip(img_bgr, 0)   # vertical flip

    # Brightness / contrast jitter
    alpha = float(rng.uniform(0.6, 1.5))
    beta  = float(rng.uniform(-25, 25))
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Rotation ±20°
    angle = float(rng.uniform(-20, 20))
    cx, cy = img_size // 2, img_size // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (img_size, img_size))

    return img_bgr


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR -> [3, H, W] float32 in [0,1]. torch.tensor() (NumPy 2.x 호환)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.tensor(rgb.transpose(2, 0, 1))  # [3, H, W]


# -- Dataset -------------------------------------------------------------------

class FireDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], img_size: int, augment: bool):
        self.paths    = paths
        self.labels   = labels
        self.img_size = img_size
        self.augment  = augment
        self._rng     = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if self.augment:
            img = _aug(img, self.img_size, self._rng)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return _to_tensor(img), self.labels[idx]


def _build_datasets(img_size: int, val_frac: float, seed: int):
    rng    = np.random.default_rng(seed)
    paths:  list[Path] = []
    labels: list[int]  = []

    for p in sorted(NORMAL_DIR.glob("*.jpg")):
        paths.append(p); labels.append(0)
    for p in sorted(FIRE_DIR.glob("*.jpg")):
        paths.append(p); labels.append(1)

    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    trn_paths  = [paths[i]  for i in trn_idx]
    trn_labels = [labels[i] for i in trn_idx]
    val_paths  = [paths[i]  for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    trn_ds = FireDataset(trn_paths, trn_labels, img_size=img_size, augment=True)
    val_ds = FireDataset(val_paths, val_labels, img_size=img_size, augment=False)
    return trn_ds, val_ds, trn_labels


def _make_sampler(labels: list[int]) -> WeightedRandomSampler:
    n0 = labels.count(0)
    n1 = labels.count(1)
    w  = [1.0 / n1 if l == 1 else 1.0 / n0 for l in labels]
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


# -- 임계값 탐색 ---------------------------------------------------------------

def _find_threshold(model: Fire2DCNN, val_loader: DataLoader, device: torch.device):
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[int]   = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            prob = torch.sigmoid(model(imgs)).cpu().tolist()
            all_probs.extend(prob if isinstance(prob, list) else [prob])
            all_labels.extend(lbls.tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    preds = (probs >= best_thr).astype(int)
    return best_thr, {
        "threshold_fire": round(best_thr, 3),
        "f1":             round(best_f1, 4),
        "accuracy":       round(float((preds == labels).mean()), 4),
        "fire_recall":    round(float(((preds==1)&(labels==1)).sum() / max((labels==1).sum(),1)), 4),
        "fire_precision": round(float(((preds==1)&(labels==1)).sum() / max((preds==1).sum(),1)), 4),
    }


# -- 학습 루프 -----------------------------------------------------------------

def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    trn_ds, val_ds, trn_labels = _build_datasets(args.img_size, args.val_frac, args.seed)
    sampler    = _make_sampler(trn_labels)
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    n0, n1 = trn_labels.count(0), trn_labels.count(1)
    print(f"[DATA] train={len(trn_ds)} (normal={n0}, fire={n1})  val={len(val_ds)}")

    model = Fire2DCNN(base_ch=args.base_ch, dropout_p=args.dropout_p).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Fire2DCNN  파라메터: {n_params:,}")

    pos_weight = torch.tensor([n0 / max(n1, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in trn_loader:
            imgs = imgs.to(device)
            lbls = lbls.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            thr, metrics = _find_threshold(model, val_loader, device)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={total_loss/len(trn_loader):.4f} | "
                f"F1={metrics['f1']:.4f} thr={thr:.2f} | "
                f"recall={metrics['fire_recall']:.4f} prec={metrics['fire_precision']:.4f}"
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "metrics": metrics, "args": vars(args)},
                           OUT_DIR / "vision_best.pt")
                with open(OUT_DIR / "vision_threshold.json", "w") as f:
                    json.dump(metrics, f, indent=2)

    print(f"\n[DONE] best F1={best_f1:.4f} -> {OUT_DIR / 'vision_best.pt'}")


# -- 진입점 --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fire2DCNN training")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
