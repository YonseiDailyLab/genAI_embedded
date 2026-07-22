import torch
import torch.nn as nn


class TinyAE(nn.Module):
    """Autoencoder for 15-sensor TinyAE system.

    Architecture  : 90-32-6-32-90  (15 features × 6 stats = 90-dim input)
    Compression   : 90/6 = 15:1  (same ratio as original 60-16-4-16-60)
    Activations   : ReLU | Linear(bottleneck) | ReLU | Sigmoid
                    Sigmoid output matches normalized [0,1] input range.
                    Linear bottleneck avoids dead-ReLU capacity loss.

    Matches AIfES Express layout for weight export:
      per layer: [out*in weights row-major, out biases]  (= nn.Linear default)
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(90, 32)
        self.enc2 = nn.Linear(32,  6)   # bottleneck — no activation
        self.dec1 = nn.Linear( 6, 32)
        self.dec2 = nn.Linear(32, 90)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc1(x))
        x = self.enc2(x)                # linear bottleneck
        x = torch.relu(self.dec1(x))
        return torch.sigmoid(self.dec2(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc2(torch.relu(self.enc1(x)))

    def init_glorot(self):
        """Xavier uniform init — matches AIfES_E_init_glorot_uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def weight_count(self) -> int:
        """Total float32 parameters (should be 6304 for 90-32-6-32-90)."""
        return sum(p.numel() for p in self.parameters())
