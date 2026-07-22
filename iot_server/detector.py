"""
이상치 검출기.

SensorDetector  : TinyAE 재구성 오차 기반 (Tukey IQR 임계값)
VisionDetector  : Fire2DCNN 화재 분류기 (vision/model.py)
                  checkpoint 없을 때 -> labeled anomaly_flag fallback (replay)
MultimodalFuser : 교차검증 로직 (한쪽 이상치 -> 다른 모달로 확인 -> 둘 다 이상치 시 확정)
"""

import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# -- 경로 설정 ------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent
PRETRAIN_DIR = REPO_ROOT / "pretrain"
LABELED_DIR  = REPO_ROOT / "data" / "labeled"
IMAGE_ROOT   = REPO_ROOT / "data" / "data"

sys.path.insert(0, str(PRETRAIN_DIR))

from model import TinyAE
from data  import window_to_stats, _load_jsonl_window
from vision.model         import Fire2DCNN
from vision.multimodal_ae import FusionAE, FUSION_DIM

_CPU              = torch.device("cpu")
_VIS_IMG_SIZE     = 112
_VIS_CKPT         = Path(__file__).parent / "vision" / "checkpoints" / "vision_best.pt"
_VIS_THR_JSON     = Path(__file__).parent / "vision" / "checkpoints" / "vision_threshold.json"
_FUSION_CKPT      = Path(__file__).parent / "vision" / "checkpoints" / "fusion_ae_best.pt"


# -- 센서 이상치 검출 ----------------------------------------------------------

class SensorDetector:
    """
    TinyAE 재구성 오차로 이상치 여부 판정.
    임계값은 정상 데이터에서 Tukey IQR으로 자동 산출 (q3 + 3·IQR).
    """

    def __init__(self, ckpt_path: Path = PRETRAIN_DIR / "tinyae_pretrained.pt"):
        self.model = TinyAE()
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model.eval()
        self._crit = nn.MSELoss(reduction="none")
        self.thr_warn, self.thr_anom = self._compute_thresholds()

    def _compute_thresholds(self) -> tuple[float, float]:
        mses: list[float] = []
        for path in sorted(LABELED_DIR.glob("TESTBED_*_labeled.jsonl")):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["metadata"].get("anomaly_flag", False):
                        continue
                    raw = _load_jsonl_window(rec)
                    vec = window_to_stats(raw)
                    x   = torch.tensor(vec).unsqueeze(0)
                    with torch.no_grad():
                        mses.append(self._crit(self.model(x), x).mean().item())
        arr = np.array(mses)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr), float(q3 + 3.0 * iqr)

    def score(self, sensor_record: dict) -> float:
        raw = _load_jsonl_window(sensor_record)
        vec = window_to_stats(raw)
        x   = torch.tensor(vec).unsqueeze(0)
        with torch.no_grad():
            return self._crit(self.model(x), x).mean().item()

    def check(self, sensor_record: dict) -> tuple[bool, float]:
        """(is_anomaly, mse_score)"""
        s = self.score(sensor_record)
        return (s > self.thr_anom), s


# -- 비전 이상치 검출 ----------------------------------------------------------

def _load_jpeg_tensor(jpeg_path: Path) -> torch.Tensor | None:
    """JPEG -> [1, 3, 112, 112] float32 in [0,1]. torch.tensor() (NumPy 2.x 호환)."""
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return None
    img = cv2.resize(img, (_VIS_IMG_SIZE, _VIS_IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]


def _frame_bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> [1, 3, 112, 112] float32."""
    img = cv2.resize(frame_bgr, (_VIS_IMG_SIZE, _VIS_IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]


class VisionDetector:
    """
    Fire2DCNN 기반 비전 이상치 검출기.

    모드 자동 선택:
      checkpoint 존재 -> Fire2DCNN 추론 (trained model)
      checkpoint 없음 -> labeled anomaly_flag fallback (replay)

    live 모드: push_frame()으로 최신 카메라 프레임 업데이트 -> check_live()
    """

    def __init__(
        self,
        ckpt_path: Path | None = _VIS_CKPT,
        thr_json:  Path | None = _VIS_THR_JSON,
    ):
        self._model: Fire2DCNN | None = None
        self._fire_thr: float = 0.5
        self._latest_frame: torch.Tensor | None = None  # live 모드용

        if ckpt_path and Path(ckpt_path).exists():
            self._load_model(Path(ckpt_path), thr_json)
        else:
            print("[VisionDetector] checkpoint 없음 -> labeled anomaly_flag fallback 사용")

    def _load_model(self, ckpt_path: Path, thr_json: Path | None):
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        args = ckpt.get("args", {})
        self._model = Fire2DCNN(
            base_ch    = args.get("base_ch", 32),
            dropout_p  = args.get("dropout_p", 0.4),
        )
        self._model.load_state_dict(ckpt["model"], strict=True)
        self._model.eval()

        if thr_json and Path(thr_json).exists():
            thr_data = json.loads(Path(thr_json).read_text())
            self._fire_thr = float(thr_data.get("threshold_fire", 0.5))

        print(f"[VisionDetector] Fire2DCNN 로드 완료 | fire_thr={self._fire_thr:.3f}")

    @property
    def has_model(self) -> bool:
        return self._model is not None

    # -- live 모드 ----------------------------------------------------------

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """카메라 최신 프레임 업데이트."""
        self._latest_frame = _frame_bgr_to_tensor(frame_bgr)

    @torch.no_grad()
    def check_live(self) -> tuple[bool, float]:
        """최신 프레임으로 추론. 프레임 없으면 (False, 0.0)."""
        if not self.has_model or self._latest_frame is None:
            return False, 0.0
        return self._infer(self._latest_frame)

    # -- 레코드(replay) 기반 ------------------------------------------------

    @torch.no_grad()
    def check(self, vision_record: dict) -> tuple[bool, float]:
        """vision record 1개로 이상치 판정."""
        if not self.has_model:
            flag = vision_record["metadata"].get("anomaly_flag", False)
            return flag, (1.0 if flag else 0.0)

        blob_uri  = vision_record.get("data", {}).get("blob_uri", "")
        tensor    = _load_jpeg_tensor(IMAGE_ROOT / blob_uri)

        if tensor is None:
            flag = vision_record["metadata"].get("anomaly_flag", False)
            return flag, 0.0

        return self._infer(tensor)

    @torch.no_grad()
    def _infer(self, x: torch.Tensor) -> tuple[bool, float]:
        """[1, 3, H, W] -> (is_fire, prob)."""
        prob = float(torch.sigmoid(self._model(x)).item())
        return (prob >= self._fire_thr), prob


# -- 멀티모달 융합 -------------------------------------------------------------

class MultimodalFuser:
    """
    Process 1 교차검증:
      1-1. 센서 이상치 -> 비전으로 교차검증 -> 둘 다 확정 시 이상치
      1-2. 비전 이상치 -> 센서로 교차검증 -> 둘 다 확정 시 이상치
      단독 이상치(한쪽만): unconfirmed -> 정상 처리
      비전 레코드 없음: 센서 단독 판정 (sensor_only)
    """

    def __init__(self, sensor: SensorDetector, vision: VisionDetector):
        self.sensor = sensor
        self.vision = vision

    def fuse(self, sensor_record: dict, vision_record: dict | None) -> dict:
        """
        Returns {
          is_anomaly    : bool
          confidence    : "multimodal" | "sensor_only" | "normal"
          disaster_type : dict
          sensor_score  : float
          sensor_thr    : float
          vision_score  : float | None
          vision_flag   : bool | None
        }
        """
        sensor_anom, sensor_score = self.sensor.check(sensor_record)
        vision_anom: bool | None  = None
        vision_score: float | None = None

        if vision_record is not None:
            vision_anom, vision_score = self.vision.check(vision_record)

        # 융합 판정
        if sensor_anom and vision_anom:
            is_anomaly = True
            confidence = "multimodal"
            dtype_src  = sensor_record
        elif sensor_anom and vision_anom is None:
            is_anomaly = True
            confidence = "sensor_only"
            dtype_src  = sensor_record
        else:
            is_anomaly = False
            confidence = "normal"
            dtype_src  = None

        disaster_type = (
            dtype_src["metadata"]["disaster_type"]
            if dtype_src else {"main_tag": "normal", "sub_tag": []}
        )

        return {
            "is_anomaly":    is_anomaly,
            "confidence":    confidence,
            "disaster_type": disaster_type,
            "sensor_score":  sensor_score,
            "sensor_thr":    self.sensor.thr_anom,
            "vision_score":  vision_score,
            "vision_flag":   vision_anom,
        }


# -- 멀티모달 AE 이상치 검출 ---------------------------------------------------

class MultimodalAEDetector:
    """
    FusionAE 기반 멀티모달 이상치 검출기.

    FusionAE checkpoint 없으면 -> MultimodalFuser(규칙 기반) fallback.

    판정 흐름:
      1. TinyAE bottleneck + Fire2DCNN GAP -> concat [262]
      2. FusionAE 재구성 오차 계산
      3. 오차 > thr_anom -> 멀티모달 이상치
      4. disaster_type은 개별 검출기 결과로 결정
    """

    def __init__(
        self,
        sensor_det: SensorDetector,
        vision_det: VisionDetector,
        ckpt_path:  Path | None = _FUSION_CKPT,
    ):
        self._sensor  = sensor_det
        self._vision  = vision_det
        self._model:  FusionAE | None = None
        self._lo:     torch.Tensor | None = None
        self._hi:     torch.Tensor | None = None
        self._thr_warn: float = 0.0
        self._thr_anom: float = 0.0
        self._fallback = MultimodalFuser(sensor_det, vision_det)

        if ckpt_path and Path(ckpt_path).exists():
            self._load(Path(ckpt_path))
        else:
            print("[MultimodalAEDetector] checkpoint 없음 -> 규칙 기반 fuser fallback 사용")

    def _load(self, ckpt_path: Path):
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        self._model = FusionAE(
            bottleneck_dim = ckpt.get("bottleneck_dim", 32),
            dropout_p      = ckpt.get("dropout_p", 0.1),
        )
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._lo       = torch.tensor(ckpt["norm_lo"])
        self._hi       = torch.tensor(ckpt["norm_hi"])
        self._thr_warn = float(ckpt["thr_warn"])
        self._thr_anom = float(ckpt["thr_anom"])
        print(f"[MultimodalAEDetector] FusionAE 로드 | thr_anom={self._thr_anom:.6f}")

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def sensor(self) -> "SensorDetector":
        """Process 1 단일 모달 트리거용 센서 검출기 접근자."""
        return self._sensor

    @property
    def vision(self) -> "VisionDetector":
        """Process 1 단일 모달 트리거용 비전 검출기 접근자."""
        return self._vision

    def _get_embeddings(
        self,
        sensor_record: dict,
        vision_record: dict | None,
    ) -> torch.Tensor | None:
        """(센서, 이미지) -> 정규화된 concat 임베딩 [1, 262]. 이미지 없으면 None."""
        if vision_record is None:
            return None

        # 센서 임베딩 (TinyAE bottleneck)
        raw = _load_jsonl_window(sensor_record)
        vec = window_to_stats(raw)
        x_s = torch.tensor(vec).unsqueeze(0)
        with torch.no_grad():
            z_s = self._sensor.model.encode(x_s).squeeze(0)  # [6]

        # 비전 임베딩 (Fire2DCNN GAP)
        blob_uri  = vision_record.get("data", {}).get("blob_uri", "")
        tensor    = _load_jpeg_tensor(IMAGE_ROOT / blob_uri)
        if tensor is None:
            return None
        with torch.no_grad():
            feat  = self._vision._model.backbone(tensor)   # [1,256,7,7]
            z_v   = self._vision._model.gap(feat).flatten()  # [256]

        # concat + 정규화
        z = torch.cat([z_s, z_v])   # [262]
        rng = (self._hi - self._lo).clamp(min=1e-8)
        z_norm = ((z - self._lo) / rng).clamp(0.0, 1.0).unsqueeze(0)  # [1,262]
        return z_norm

    @torch.no_grad()
    def fuse(self, sensor_record: dict, vision_record: dict | None) -> dict:
        """규칙 기반 fuser와 동일한 인터페이스."""
        # FusionAE 없으면 규칙 기반 fallback
        if not self.has_model or not self._vision.has_model:
            return self._fallback.fuse(sensor_record, vision_record)

        z_norm = self._get_embeddings(sensor_record, vision_record)

        # 임베딩 추출 실패 -> 센서 단독 판정
        if z_norm is None:
            sensor_anom, sensor_score = self._sensor.check(sensor_record)
            return {
                "is_anomaly":    sensor_anom,
                "confidence":    "sensor_only" if sensor_anom else "normal",
                "disaster_type": sensor_record["metadata"]["disaster_type"] if sensor_anom
                                 else {"main_tag": "normal", "sub_tag": []},
                "sensor_score":  sensor_score,
                "sensor_thr":    self._sensor.thr_anom,
                "fusion_score":  None,
                "fusion_thr":    self._thr_anom,
                "vision_score":  None,
                "vision_flag":   None,
            }

        # FusionAE 재구성 오차
        z_hat        = self._model(z_norm)
        fusion_score = float((z_hat - z_norm).pow(2).mean().item())
        is_anomaly   = fusion_score > self._thr_anom

        # disaster_type: 개별 검출기에서 결정
        _, sensor_score = self._sensor.check(sensor_record)
        vision_anom, vision_score = self._vision.check(vision_record) if vision_record else (None, None)

        if is_anomaly:
            # 어느 모달에서 이상치 신호가 왔는지 확인해 타입 결정
            dtype_src = sensor_record if sensor_score > self._sensor.thr_anom else None
            if dtype_src is None and vision_anom:
                dtype_src = vision_record
            disaster_type = (
                dtype_src["metadata"]["disaster_type"]
                if dtype_src else {"main_tag": "anomaly", "sub_tag": []}
            )
            confidence = "multimodal_ae"
        else:
            disaster_type = {"main_tag": "normal", "sub_tag": []}
            confidence    = "normal"

        return {
            "is_anomaly":    is_anomaly,
            "confidence":    confidence,
            "disaster_type": disaster_type,
            "sensor_score":  sensor_score,
            "sensor_thr":    self._sensor.thr_anom,
            "fusion_score":  fusion_score,
            "fusion_thr":    self._thr_anom,
            "vision_score":  vision_score,
            "vision_flag":   vision_anom,
        }
