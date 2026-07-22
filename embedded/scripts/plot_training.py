#!/usr/bin/env python3
"""
TinyAE Training & Inference Visualizer
Usage:
  # SD 카드 CSV (기본)
  python plot_training.py --sd /Volumes/SD/datas

  # 시리얼 로그 파일
  python plot_training.py --log /tmp/tinyae_log3.txt

  # MQTT 실시간 (별도 터미널에서 mosquitto_sub로 파이프)
  mosquitto_sub -h 192.168.0.23 -t TinyAE/train | python plot_training.py --mqtt
"""

import argparse
import re
import sys
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 PNG 저장 가능
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_log_file(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse serial log → (train_df, infer_df)."""
    train_rows, infer_rows = [], []

    # ESP-IDF log line: "I (TICK) TAG: message"
    re_tick   = re.compile(r'[IWE] \((\d+)\) TinyAE_Pipe: (.+)')
    re_train  = re.compile(
        r'Train round=(\d+) -> v(\d+) mse=([\d.]+) q7=(\d)')
    re_infer_thr = re.compile(
        r'Infer\((\w+)\) v(\d+) mse=([\d.]+) warn=([\d.]+) anom=([\d.]+) \[(\w+)\]')
    re_infer_base = re.compile(
        r'Infer\((\w+)\) v(\d+) mse=([\d.]+) \(baseline (\d+)/(\d+)\)')

    with open(path, errors='replace') as f:
        for line in f:
            m = re_tick.search(line)
            if not m:
                continue
            tick_ms = int(m.group(1))
            msg     = m.group(2)

            mt = re_train.search(msg)
            if mt:
                train_rows.append({
                    'tick_s': tick_ms / 1000,
                    'round':  int(mt.group(1)),
                    'version': int(mt.group(2)),
                    'mse':    float(mt.group(3)),
                    'q7':     int(mt.group(4)),
                })

            mi = re_infer_thr.search(msg)
            if mi:
                infer_rows.append({
                    'tick_s':   tick_ms / 1000,
                    'mode':     mi.group(1),
                    'version':  int(mi.group(2)),
                    'mse':      float(mi.group(3)),
                    'thr_warn': float(mi.group(4)),
                    'thr_anom': float(mi.group(5)),
                    'severity': mi.group(6),
                    'baseline': None,
                })

            mb = re_infer_base.search(msg)
            if mb:
                infer_rows.append({
                    'tick_s':   tick_ms / 1000,
                    'mode':     mb.group(1),
                    'version':  int(mb.group(2)),
                    'mse':      float(mb.group(3)),
                    'thr_warn': float('nan'),
                    'thr_anom': float('nan'),
                    'severity': 'baseline',
                    'baseline': int(mb.group(4)),
                })

    return pd.DataFrame(train_rows), pd.DataFrame(infer_rows)


def parse_sd_csvs(sd_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read tYYMMDD.csv and iYYMMDD.csv from SD card mount point."""
    base = Path(sd_path)

    train_files = sorted(base.glob("t??????.csv"))
    infer_files = sorted(base.glob("i??????.csv"))

    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True) \
               if train_files else pd.DataFrame()
    infer_df = pd.concat([pd.read_csv(f) for f in infer_files], ignore_index=True) \
               if infer_files else pd.DataFrame()

    if not train_df.empty:
        train_df['tick_s'] = train_df['timestamp'] - train_df['timestamp'].iloc[0]
    if not infer_df.empty:
        infer_df['tick_s'] = infer_df['timestamp'] - infer_df['timestamp'].iloc[0]
        infer_df['thr_warn'] = pd.to_numeric(infer_df['thr_warn'], errors='coerce')
        infer_df['thr_anom'] = pd.to_numeric(infer_df['thr_anom'], errors='coerce')

    return train_df, infer_df


def read_mqtt_stdin() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read JSON lines from stdin (mosquitto_sub piped in)."""
    train_rows = []
    print("Reading MQTT from stdin (Ctrl+C to stop)...", file=sys.stderr)
    for line in sys.stdin:
        try:
            d = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if 'round' in d:
            train_rows.append(d)
    df = pd.DataFrame(train_rows)
    return df, pd.DataFrame()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(train_df: pd.DataFrame, infer_df: pd.DataFrame, title: str = "TinyAE"):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── Panel 1: Training MSE over time ──────────────────────────────────
    if not train_df.empty and 'mse' in train_df:
        ax1.semilogy(train_df['tick_s'] / 60, train_df['mse'],
                     color='royalblue', lw=1.2, alpha=0.8, label='Train MSE')
        ax1.axhline(5e-4, color='tomato', ls='--', lw=1, label='target (5e-4)')
        ax1.set_ylabel('MSE (log scale)')
        ax1.set_title('Training Loss')
        ax1.legend(fontsize=8)
        ax1.grid(True, which='both', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No training data', ha='center', va='center',
                 transform=ax1.transAxes)

    # ── Panel 2: Inference MSE + IQR thresholds ──────────────────────────
    if not infer_df.empty and 'mse' in infer_df:
        t = infer_df['tick_s'] / 60

        normal_mask  = infer_df['severity'] == 'normal'
        warn_mask    = infer_df['severity'] == 'warning'
        anom_mask    = infer_df['severity'].isin(['anomaly', 'severe'])
        base_mask    = infer_df['severity'] == 'baseline'

        ax2.scatter(t[base_mask],  infer_df.loc[base_mask, 'mse'],
                    s=10, color='gray',   alpha=0.5, label='building baseline', zorder=2)
        ax2.scatter(t[normal_mask], infer_df.loc[normal_mask, 'mse'],
                    s=15, color='steelblue', alpha=0.7, label='normal', zorder=3)
        ax2.scatter(t[warn_mask],  infer_df.loc[warn_mask, 'mse'],
                    s=30, color='orange', alpha=0.9, label='warning', zorder=4)
        ax2.scatter(t[anom_mask],  infer_df.loc[anom_mask, 'mse'],
                    s=50, color='red',    alpha=0.9, label='anomaly/severe', zorder=5, marker='x')

        # IQR threshold bands
        thr_warn = infer_df['thr_warn'].dropna()
        thr_anom = infer_df['thr_anom'].dropna()
        if not thr_warn.empty:
            t_valid = t[thr_warn.index]
            ax2.plot(t_valid, thr_warn.values,
                     color='orange', ls='--', lw=1.2, label='warn (Q3+1.5IQR)')
            ax2.plot(t_valid, thr_anom.values,
                     color='red',    ls='--', lw=1.2, label='anom (Q3+3.0IQR)')
            ax2.fill_between(t_valid, thr_warn.values, thr_anom.values,
                             alpha=0.08, color='orange')
            ax2.fill_between(t_valid, thr_anom.values,
                             infer_df.loc[thr_anom.index, 'mse'].max() * 1.5,
                             alpha=0.06, color='red')

        ax2.set_ylabel('MSE')
        ax2.set_title('Inference MSE & IQR Thresholds')
        ax2.legend(fontsize=7, ncol=3)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No inference data', ha='center', va='center',
                 transform=ax2.transAxes)

    # ── Panel 3: Anomaly timeline ─────────────────────────────────────────
    if not infer_df.empty and 'severity' in infer_df:
        sev_map = {'normal': 0, 'baseline': 0, 'warning': 1, 'anomaly': 2, 'severe': 3}
        sev_num = infer_df['severity'].map(sev_map).fillna(0)
        colors  = sev_num.map({0: 'steelblue', 1: 'orange', 2: 'red', 3: 'darkred'})
        ax3.bar(infer_df['tick_s'] / 60, sev_num + 0.8,
                width=0.8, color=colors, alpha=0.8, bottom=-0.4)
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['normal', 'warn', 'anom', 'severe'], fontsize=8)
        ax3.set_ylabel('Severity')
        ax3.set_title('Anomaly Timeline')
        ax3.grid(True, axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No severity data', ha='center', va='center',
                 transform=ax3.transAxes)

    ax3.set_xlabel('Elapsed time (min)')

    plt.savefig('tinyae_training.png', dpi=150, bbox_inches='tight')
    print("Saved: tinyae_training.png")
    try:
        plt.show()
    except Exception:
        pass  # headless 환경에서 show() 실패해도 PNG는 저장됨


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TinyAE Training Visualizer")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--log', metavar='FILE',
                     help='Serial log file (from cat /dev/cu.usbmodemXXXX)')
    src.add_argument('--sd',  metavar='PATH',
                     help='SD card datas/ directory (e.g. /Volumes/SD/datas)')
    src.add_argument('--mqtt', action='store_true',
                     help='Read JSON from stdin (mosquitto_sub piped)')
    ap.add_argument('--title', default='TinyAE Training', help='Plot title')
    args = ap.parse_args()

    if args.log:
        print(f"Parsing log: {args.log}")
        train_df, infer_df = parse_log_file(args.log)
        print(f"  Train rows: {len(train_df)}, Infer rows: {len(infer_df)}")
    elif args.sd:
        print(f"Reading SD CSVs from: {args.sd}")
        train_df, infer_df = parse_sd_csvs(args.sd)
        print(f"  Train rows: {len(train_df)}, Infer rows: {len(infer_df)}")
    else:
        train_df, infer_df = read_mqtt_stdin()

    plot(train_df, infer_df, title=args.title)


if __name__ == '__main__':
    main()
