"""
FusionAE: 멀티모달 이상치 탐지용 임베딩 공간 오토인코더.

입력: TinyAE bottleneck [6] + Fire2DCNN GAP features [256] -> concat [262]
구조: 262 -> 128 -> 32 (bottleneck) -> 128 -> 262
학습: 정상 이벤트만 사용 (비지도)
판정: 재구성 오차 > Tukey IQR 임계값 -> 이상치
"""

import torch
import torch.nn as nn

SENSOR_DIM = 6    # TinyAE bottleneck 차원
VISION_DIM = 256  # Fire2DCNN GAP 차원 (base_ch=32 -> 32*8=256)
FUSION_DIM = SENSOR_DIM + VISION_DIM  # 262


class FusionAE(nn.Module):
    """
    정상 (센서 임베딩, 이미지 임베딩) 쌍의 분포를 학습.
    이상치 발생 시 두 모달리티 간 정상 상관관계가 무너져 재구성 오차 증가.

    Input/Output: [B, 262] float32 (정규화된 임베딩)
    """

    def __init__(self, bottleneck_dim: int = 32, dropout_p: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(FUSION_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, FUSION_DIM),
            nn.Sigmoid(),   # 입력이 [0,1]로 정규화됐을 때 사용
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 262] -> [B, 262]"""
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
