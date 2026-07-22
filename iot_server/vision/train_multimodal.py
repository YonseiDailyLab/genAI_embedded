"""
FusionAE 학습 스크립트.

1. TinyAE + Fire2DCNN을 frozen feature extractor로 사용
2. 정상 이벤트의 (센서, 이미지) 쌍에서 임베딩 pre-extract
3. FusionAE를 재구성 오차(MSE) 기준으로 학습
4. 임계값: 정상 검증셋의 Tukey IQR (q3 + 3·IQR)

Usage:
  cd iot_server && python -m vision.train_multimodal
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT    = Path(__file__).parent.parent.parent
PRETRAIN_DIR = REPO_ROOT / "pretrain"
LABELED_DIR  = REPO_ROOT / "data" / "labeled"
VISION_JSONL = REPO_ROOT / "data" / "data" / "vision_labeled.jsonl"
IMAGE_ROOT   = REPO_ROOT / "data" / "data"
OUT_DIR      = Path(__file__).parent / "checkpoints"

sys.path.insert(0, str(PRETRAIN_DIR))
from model import TinyAE
from data  import window_to_stats, _load_jsonl_window

from vision.model         import Fire2DCNN
from vision.multimodal_ae import FusionAE, FUSION_DIM

_IMG_SIZE = 112

# -- 학습 기본값 ----------------------------------------------------------------

DEFAULTS = {
    "epochs":        150,
    "lr":            1e-3,
    "batch_size":    64,
    "bottleneck_dim": 32,
    "dropout_p":     0.1,
    "val_frac":      0.15,
    "seed":          42,
}


# -- Feature Extractor ---------------------------------------------------------

class FeatureExtractor:
    """TinyAE(sensor) + Fire2DCNN(vision) -> frozen 임베딩 추출."""

    def __init__(self):
        # CNN forward(3만 장)는 GPU에서 훨씬 빠름 -> 추출 단계만 CUDA 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  FeatureExtractor device: {self.device}")

        # TinyAE
        self.tinyae = TinyAE()
        ckpt = torch.load(str(PRETRAIN_DIR / "tinyae_pretrained.pt"), map_location="cpu")
        self.tinyae.load_state_dict(ckpt)
        self.tinyae.eval().to(self.device)

        # Fire2DCNN
        vis_ckpt_path = OUT_DIR / "vision_best.pt"
        ckpt = torch.load(str(vis_ckpt_path), map_location="cpu")
        args = ckpt.get("args", {})
        self.fire_cnn = Fire2DCNN(
            base_ch   = args.get("base_ch", 32),
            dropout_p = 0.0,
        )
        self.fire_cnn.load_state_dict(ckpt["model"], strict=True)
        self.fire_cnn.eval().to(self.device)

    @torch.no_grad()
    def sensor_embed(self, sensor_record: dict) -> torch.Tensor:
        """센서 레코드 -> TinyAE bottleneck [6]."""
        raw = _load_jsonl_window(sensor_record)
        vec = window_to_stats(raw)
        x   = torch.tensor(vec).unsqueeze(0).to(self.device)
        return self.tinyae.encode(x).squeeze(0).cpu()   # [6]

    @torch.no_grad()
    def vision_embed(self, blob_uri: str) -> torch.Tensor | None:
        """이미지 -> Fire2DCNN GAP features [256]."""
        img = cv2.imread(str(IMAGE_ROOT / blob_uri))
        if img is None:
            return None
        img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x   = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)  # [1,3,H,W]
        feat = self.fire_cnn.backbone(x)    # [1, 256, 7, 7]
        feat = self.fire_cnn.gap(feat)      # [1, 256, 1, 1]
        return feat.flatten().cpu()         # [256]


# -- 데이터 준비 ---------------------------------------------------------------

def _build_embeddings(extractor: FeatureExtractor) -> torch.Tensor:
    """
    정상 이벤트의 (sensor, vision) 쌍에서 임베딩 추출.
    Returns: [N, 262] float32
    """
    # vision JSONL -> {event_id: blob_uri} (정상만)
    vision_map: dict[str, str] = {}
    with open(VISION_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            if not rec["metadata"]["anomaly_flag"]:
                vision_map[rec["event_id"]] = rec["data"]["blob_uri"]

    embeddings: list[torch.Tensor] = []
    ok = skip = 0

    for jsonl_path in sorted(LABELED_DIR.glob("TESTBED_*_labeled.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["metadata"]["anomaly_flag"]:
                    continue

                event_id = rec.get("event_id")
                if event_id not in vision_map:
                    skip += 1
                    continue

                z_s = extractor.sensor_embed(rec)
                z_v = extractor.vision_embed(vision_map[event_id])
                if z_v is None:
                    skip += 1
                    continue

                embeddings.append(torch.cat([z_s, z_v]))  # [262]
                ok += 1

    print(f"  임베딩 추출: {ok}건 완료, {skip}건 스킵")
    return torch.stack(embeddings)  # [N, 262]


def _normalize(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max 정규화 -> [0,1]. (Sigmoid 출력과 맞추기 위해)"""
    lo  = data.min(dim=0).values
    hi  = data.max(dim=0).values
    rng = (hi - lo).clamp(min=1e-8)
    return (data - lo) / rng, lo, hi


def _tukey_threshold(normal_mse: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(normal_mse, 25), np.percentile(normal_mse, 75)
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr), float(q3 + 3.0 * iqr)


# -- 학습 루프 -----------------------------------------------------------------

def train(epochs=150, lr=1e-3, batch_size=64,
          bottleneck_dim=32, dropout_p=0.1,
          val_frac=0.15, seed=42):

    torch.manual_seed(seed)
    rng    = np.random.default_rng(seed)
    device = torch.device("cpu")   # AE는 작으니 CPU로 충분

    # 임베딩 캐시 (1만+ 장 CPU 추출은 ~10분 소요 -> 한번 추출하면 재사용)
    cache_path = OUT_DIR / "fusion_embeddings.pt"
    if cache_path.exists():
        print(f"[1-2/3] 임베딩 캐시 로드: {cache_path}")
        data = torch.load(str(cache_path))
        print(f"  전체 {len(data)}건, 차원={data.shape[1]}")
    else:
        print("[1/3] Feature extractor 로드...")
        extractor = FeatureExtractor()
        print("[2/3] 정상 이벤트 임베딩 추출 중...")
        data = _build_embeddings(extractor)
        print(f"  전체 {len(data)}건, 차원={data.shape[1]}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(cache_path))
        print(f"  임베딩 캐시 저장: {cache_path}")

    # 정규화
    data_norm, lo, hi = _normalize(data)

    # train / val 분할 (torch 인덱싱 — NumPy 2.x 호환 위해 torch.long 텐서 사용)
    idx   = torch.tensor(rng.permutation(len(data_norm)), dtype=torch.long)
    n_val = max(1, int(len(idx) * val_frac))
    val_data = data_norm[idx[:n_val]]
    trn_data = data_norm[idx[n_val:]]
    print(f"  train={len(trn_data)}, val={len(val_data)}")

    trn_loader = DataLoader(TensorDataset(trn_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data),  batch_size=batch_size, shuffle=False)

    # 모델
    model    = FusionAE(bottleneck_dim=bottleneck_dim, dropout_p=dropout_p).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/3] FusionAE 학습 시작 | 파라메터: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        trn_loss = 0.0
        for (x,) in trn_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_mses: list[float] = []
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(device)
                    err = (model(x) - x).pow(2).mean(dim=1)
                    val_mses.extend(err.tolist())

            val_arr   = np.array(val_mses)
            thr_warn, thr_anom = _tukey_threshold(val_arr)

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_loss={trn_loss/len(trn_loader):.6f} | "
                f"val_mse={val_arr.mean():.6f} | "
                f"thr_anom={thr_anom:.6f}"
            )

            if val_arr.mean() < best_val_mse:
                best_val_mse = val_arr.mean()
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model":          model.state_dict(),
                    "epoch":          epoch,
                    "bottleneck_dim": bottleneck_dim,
                    "dropout_p":      dropout_p,
                    "norm_lo":        lo.tolist(),
                    "norm_hi":        hi.tolist(),
                    "thr_warn":       thr_warn,
                    "thr_anom":       thr_anom,
                    "val_mse":        float(val_arr.mean()),
                }, OUT_DIR / "fusion_ae_best.pt")

    print(f"\n[DONE] best val_mse={best_val_mse:.6f} -> {OUT_DIR/'fusion_ae_best.pt'}")


# -- 진입점 --------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("FusionAE training")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    train(**vars(args))
