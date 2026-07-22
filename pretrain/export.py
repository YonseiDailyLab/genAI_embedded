"""Export pretrained PyTorch weights to AIfES flat-binary format.

Output format mirrors sd_save_model() / sd_load_model() in pipeline.cpp:
  [0:4]  magic      0x54414503  ('TAE\\x03', little-endian uint32)
  [4:8]  weight_cnt             (little-endian uint32)
  [8:]   weights    float32 LE  (weight_cnt × 4 bytes)

AIfES weight layout per dense layer [in→out]:
  out*in float32  (row-major: out neurons, each with in weights)
  out    float32  (biases)

nn.Linear(in, out).weight has shape [out, in] and .flatten() gives the same
row-major order — so the export is a direct concatenation.

Expected total for 60-16-4-16-60:
  60×16+16 = 976
  16×4+4   = 68
  4×16+16  = 80
  16×60+60 = 1020
  ─────────────
  total    = 2144 weights  →  8 + 2144×4 = 8592 bytes
"""

import argparse
import struct
from pathlib import Path

import torch

from model import TinyAE

_MAGIC = 0x54414503  # kModelFileMagic in pipeline.cpp
_EXPECTED_WEIGHTS = 6304   # 90-32-6-32-90: (90×32+32)+(32×6+6)+(6×32+32)+(32×90+90)


def export_aifes_bin(model: TinyAE, output_path: str) -> int:
    model.eval()

    flat: list[float] = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            flat.extend(layer.weight.data.cpu().float().numpy().flatten().tolist())
            flat.extend(layer.bias.data.cpu().float().numpy().tolist())

    weight_count = len(flat)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", _MAGIC))
        f.write(struct.pack("<I", weight_count))
        for w in flat:
            f.write(struct.pack("<f", w))

    size = Path(output_path).stat().st_size
    status = "OK" if weight_count == _EXPECTED_WEIGHTS else f"WARNING expected {_EXPECTED_WEIGHTS}"
    print(f"exported  {weight_count} weights  {size} bytes  [{status}]  →  {output_path}")
    return weight_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TinyAE weights to AIfES binary")
    parser.add_argument("--weights", default="tinyae_pretrained.pt",
                        help="Input: PyTorch state-dict (.pt)")
    parser.add_argument("--output",  default="ae_model.bin",
                        help="Output: AIfES binary (.bin) — copy to /models/ on SD card")
    args = parser.parse_args()

    model = TinyAE()
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True))
    export_aifes_bin(model, args.output)
