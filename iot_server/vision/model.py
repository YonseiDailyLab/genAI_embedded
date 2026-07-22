"""
Fire2DCNN: 정적 이미지 기반 화재 이진 분류 모델.

단일 JPEG 프레임 -> [1, 3, H, W] -> 화재 확률

구조:
  Conv2D stem -> 4 stage downsampling -> GAP -> FC classifier
  Jetson 메모리 제약 고려: base_channels=32, 입력 112×112
"""

import torch
import torch.nn as nn


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Fire2DCNN(nn.Module):
    """
    Input:  [B, 3, H, W] float32 in [0,1] (H=W=112 권장)
    Output: logit [B]  (BCEWithLogitsLoss용, sigmoid 적용 전)

    base_ch=32 기준 파라메터 수: ~280K
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32, dropout_p: float = 0.4):
        super().__init__()
        c = base_ch

        self.backbone = nn.Sequential(
            _ConvBnRelu(in_channels, c,     stride=2),   # 112 -> 56
            _ConvBnRelu(c,           c * 2, stride=2),   # 56  -> 28
            _ConvBnRelu(c * 2,       c * 4, stride=2),   # 28  -> 14
            _ConvBnRelu(c * 4,       c * 8, stride=2),   # 14  -> 7
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, c*8, 1, 1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(c * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit [B]."""
        feat = self.backbone(x)
        feat = self.gap(feat)
        return self.head(feat).squeeze(1)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid 확률 [B] 반환."""
        return torch.sigmoid(self.forward(x))
