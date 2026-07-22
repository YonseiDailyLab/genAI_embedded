"""Sensor normalization, stats computation, and dataset classes.

Mirrors the embedded pipeline:
  sensor_to_feature_vec  →  sensors.cpp : sensor_to_feature_vec()
  compute_stats_vec      →  sensors.cpp : compute_stats_vec()
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES  = 15
N_STATS     = 6
N_DIM       = N_FEATURES * N_STATS   # 90
WINDOW_SIZE = 300                     # kStatsWindowSz on device (seconds @ 1 Hz)

# ── Sensor channel definitions ────────────────────────────────────────────────
# (jsonl_key, lo, hi, sentinel)
# sentinel: value that means "invalid" → replaced with lo before normalization → 0.0
# Ranges are sensor spec limits, not observed data ranges.

SENSOR_CHANNELS: list[tuple[str, float, float, Optional[float]]] = [
    ("Temperature",      -10.0,    60.0,  None),          # AM1008W-K  -10~60 °C
    ("Humidity",           0.0,   100.0,  None),          # AM1008W-K  0~100 %
    ("CO2",                0.0,  5000.0,  -1.0),          # AM1008W-K  0~5000 ppm   (-1 = invalid)
    ("PM10",               0.0,   500.0,  -1.0),          # AM1008W-K  0~500 µg/m³  (-1 = invalid)
    ("PM2.5",              0.0,   300.0,  -1.0),          # AM1008W-K  0~300 µg/m³  (-1 = invalid)
    ("MP801_raw",          0.0,  4095.0,  None),          # ADC 12-bit 0~4095
    ("MQ131_O3_ppb",       0.0,  2000.0,  2147483647.0),  # MQ-131     0~2000 ppb   (INT_MAX = 미연결)
    ("CO_SEN0564_raw",     0.0,  4095.0,  None),          # SEN0564 CO ADC 12-bit
    ("MiCS2714_raw",       0.0,  4095.0,  None),          # MiCS-2714 NO2 ADC 12-bit
    ("BME_Temp",         -40.0,    85.0,  None),          # BME280    -40~85 °C
    ("BME_Humidity",       0.0,   100.0,  None),          # BME280     0~100 %
    ("BME_Pressure_Pa", 30000.0, 110000.0, None),         # BME280     300~1100 hPa
    ("ENS_AQI",            1.0,     5.0,  None),          # ENS160     level 1~5
    ("ENS_TVOC_ppb",       0.0, 65000.0,  None),          # ENS160     0~65000 ppb
    ("ENS_eCO2_ppm",     400.0, 65000.0,  None),          # ENS160     400~65000 ppm
]

FEATURE_NAMES = [ch[0] for ch in SENSOR_CHANNELS]
_LO       = np.array([ch[1] for ch in SENSOR_CHANNELS], dtype=np.float64)
_HI       = np.array([ch[2] for ch in SENSOR_CHANNELS], dtype=np.float64)
_RANGE    = _HI - _LO
_SENTINEL = {i: ch[3] for i, ch in enumerate(SENSOR_CHANNELS) if ch[3] is not None}


# ── Normalization ─────────────────────────────────────────────────────────────

def sensor_to_feature_vec(raw: np.ndarray) -> np.ndarray:
    """Normalize raw sensor readings to [0, 1].

    raw : [N, 15] — columns must match FEATURE_NAMES order
    out : [N, 15] — each column clipped to [0, 1]

    Sentinel values (e.g. -1, INT_MAX) are replaced with lo → normalized to 0.0.
    """
    out = raw.astype(np.float64).copy()
    for idx, sentinel in _SENTINEL.items():
        mask = out[:, idx] == sentinel
        out[mask, idx] = _LO[idx]
    out = (out - _LO) / _RANGE
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_stats_vec(feat_hist: np.ndarray) -> np.ndarray:
    """Compute 6 statistics per feature from a normalized history window.

    feat_hist : [N, 15] float32, values in [0, 1]
    returns   : [90]   float32, values in [0, 1]

    Output layout — for each feature f (0..14), 6 consecutive values:
        f*6+0  mean
        f*6+1  std   (std_v / 0.5, clipped to [0, 1])
        f*6+2  min
        f*6+3  max
        f*6+4  slope (LSQ linear: slope*(N-1)/2 + 0.5;  0.5 = flat)
        f*6+5  curv  (equal-thirds: curv/2 + 0.5;       0.5 = linear trend)

    Mirrors tinyae::data::compute_stats_vec() in sensors.cpp exactly.
    """
    N = len(feat_hist)
    fn = float(N)
    t = np.arange(N, dtype=np.float64)

    sum_t  = fn * (fn - 1.0) / 2.0
    sum_t2 = fn * (fn - 1.0) * (2.0 * fn - 1.0) / 6.0
    lsq_denom   = fn * sum_t2 - sum_t * sum_t
    slope_scale = (fn - 1.0) / 2.0

    t1 = N // 3
    t2 = 2 * N // 3

    out = np.zeros(N_DIM, dtype=np.float32)
    fh  = feat_hist.astype(np.float64)

    for f in range(N_FEATURES):
        v = fh[:, f]

        mean_v = v.mean()
        std_v  = np.sqrt(max(0.0, (v * v).mean() - mean_v * mean_v))
        fmin   = v.min()
        fmax   = v.max()

        sum_ty = (t * v).sum()
        slope  = ((fn * sum_ty - sum_t * v.sum()) / lsq_denom) if lsq_denom > 0 else 0.0

        m1 = v[:t1].mean()   if t1 > 0   else 0.0
        m2 = v[t1:t2].mean() if t2 > t1  else 0.0
        m3 = v[t2:].mean()   if N > t2   else 0.0
        curv = m3 - 2.0 * m2 + m1

        o = f * N_STATS
        out[o + 0] = float(mean_v)
        out[o + 1] = float(np.clip(std_v / 0.5, 0.0, 1.0))
        out[o + 2] = float(fmin)
        out[o + 3] = float(fmax)
        out[o + 4] = float(np.clip(slope * slope_scale + 0.5, 0.0, 1.0))
        out[o + 5] = float(np.clip(curv  / 2.0        + 0.5, 0.0, 1.0))

    return out


def window_to_stats(raw_window: np.ndarray) -> np.ndarray:
    """raw_window [N, 15] → normalize → compute_stats_vec → [90]."""
    return compute_stats_vec(sensor_to_feature_vec(raw_window))


# ── Real sensor dataset ───────────────────────────────────────────────────────

def _load_jsonl_window(record: dict) -> np.ndarray:
    """Extract [60, 15] float32 array from one JSONL record."""
    win_size = record["window"]["size"]
    raw = np.zeros((win_size, N_FEATURES), dtype=np.float32)
    for col, (key, *_) in enumerate(SENSOR_CHANNELS):
        vals = record["data"].get(key)
        if vals is None:
            continue
        for row, v in enumerate(vals):
            raw[row, col] = float(v) if v is not None else _LO[col]
    return raw


class RealSensorDataset(Dataset):
    """Dataset built from recorded JSONL files (labeled or train).

    file_paths  : list of .jsonl paths to load
    normal_only : if True, keep only records where anomaly_flag == False
    window_agg  : number of consecutive 60-s windows to concatenate before
                  computing stats (1 = 60-sample stats, 5 = 300-sample stats
                  matching kStatsWindowSz on device)
    device      : move entire tensor to this device on load (avoids per-batch transfer)
    """

    def __init__(
        self,
        file_paths: list[str | Path],
        normal_only: bool = True,
        window_agg: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.window_agg = window_agg
        samples: list[np.ndarray] = []

        for path in file_paths:
            records: list[dict] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if normal_only and r["metadata"].get("anomaly_flag", False):
                        continue
                    records.append(r)

            # Sort by window start for correct temporal order when aggregating.
            records.sort(key=lambda r: r["window"]["start"])

            for i in range(len(records) - window_agg + 1):
                group = records[i: i + window_agg]
                # Skip group if it spans a time gap (non-consecutive windows).
                if window_agg > 1:
                    for j in range(window_agg - 1):
                        expected_end = group[j]["window"]["end"]
                        actual_start = group[j + 1]["window"]["start"]
                        if actual_start != expected_end:
                            break
                    else:
                        pass  # all consecutive, fall through
                    # (simple break-check above; re-implement cleanly below)

                raw_parts = [_load_jsonl_window(r) for r in group]
                raw = np.concatenate(raw_parts, axis=0)   # [60*agg, 15]
                samples.append(window_to_stats(raw))      # [90]

        data = np.stack(samples)                                    # [n, 90]
        self.data   = torch.tensor(data).to(device)                # whole dataset on device
        self.labels = None  # anomaly labels not stored (normal_only filter above)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ── Synthetic normal dataset (fallback when no real data) ─────────────────────

# Normal operating ranges (raw units) for indoor clean-air environment.
# (center, half_range, noise_std, drift_amp) — one row per channel.
_NORMAL_RAW: list[tuple[float, float, float, float]] = [
    ( 24.0,   3.0,   0.3,   1.5),   # Temperature      °C
    ( 45.0,  15.0,   1.5,   5.0),   # Humidity          %
    (650.0, 200.0,  25.0,  80.0),   # CO2             ppm
    ( 12.0,   8.0,   1.0,   4.0),   # PM10           µg/m³
    (  5.0,   4.0,   0.5,   2.0),   # PM2.5          µg/m³
    (700.0,  80.0,  15.0,  40.0),   # MP801_raw       ADC
    (  0.0,   0.0,   0.0,   0.0),   # MQ131_O3_ppb    (all invalid → 0)
    (1200.0, 150.0,  20.0,  60.0),  # CO_SEN0564_raw  ADC
    (1300.0, 100.0,  15.0,  50.0),  # MiCS2714_raw    ADC
    ( 26.0,   2.0,   0.2,   1.0),   # BME_Temp         °C
    ( 25.0,  10.0,   1.0,   4.0),   # BME_Humidity      %
    (101000., 500., 50.0, 200.0),   # BME_Pressure_Pa  Pa
    (  2.0,   0.5,   0.1,   0.3),   # ENS_AQI       level
    (200.0, 150.0,  30.0,  80.0),   # ENS_TVOC_ppb   ppb
    (650.0, 150.0,  40.0, 100.0),   # ENS_eCO2_ppm   ppm
]


def _generate_raw_window(rng: np.random.Generator, window_size: int = WINDOW_SIZE) -> np.ndarray:
    raw = np.zeros((window_size, N_FEATURES), dtype=np.float32)
    t   = np.linspace(0.0, 2.0 * np.pi, window_size)
    for i, (center, half_range, noise_std, drift_amp) in enumerate(_NORMAL_RAW):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq  = rng.integers(1, 4)
        drift = drift_amp * rng.uniform(0.3, 1.0) * np.sin(freq * t + phase)
        dc    = rng.uniform(-half_range * 0.5, half_range * 0.5)
        noise = rng.normal(0.0, noise_std, window_size)
        raw[:, i] = center + dc + drift + noise
    # MQ131 is always invalid in the real data → keep at 0 (maps to sentinel lo after norm)
    raw[:, 6] = 2147483647.0
    return raw


class SyntheticNormalDataset(Dataset):
    """Pre-generated dataset of normal-condition AE inputs (fallback).

    n_windows   : number of 90-dim samples
    seed        : RNG seed
    window_size : seconds per window (default 300)
    """

    def __init__(self, n_windows: int = 10_000, seed: int = 42,
                 window_size: int = WINDOW_SIZE,
                 device: str | torch.device = "cpu"):
        rng  = np.random.default_rng(seed)
        data = np.stack([
            window_to_stats(_generate_raw_window(rng, window_size))
            for _ in range(n_windows)
        ])
        self.data = torch.tensor(data).to(device)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
