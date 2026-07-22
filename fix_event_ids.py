"""Fix duplicate event_ids in labeled sensor JSONL files.

Old: event_id was incorrectly shared across different time windows.
New: event_id = evt_{YYYYMMDD}_{window_start_unix}  (guaranteed unique per window)

Updates both data/labeled/ and data/data/ sensor files in-place.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

LABELED_DIR = Path("data/labeled")
DATA_DIR    = Path("data/data")


def new_event_id(window_start: int) -> str:
    bucket = (window_start // 60) * 60   # floor to nearest 60-s boundary
    dt = datetime.fromtimestamp(bucket, tz=timezone.utc)
    return f"evt_{dt.strftime('%Y%m%d')}_{bucket}"


def fix_file(src: Path, dst: Path):
    lines = [l for l in open(src) if l.strip()]
    out = []
    for line in tqdm(lines, desc=src.name, unit="rec", leave=False):
        r = json.loads(line)
        r["event_id"] = new_event_id(r["window"]["start"])
        out.append(json.dumps(r, ensure_ascii=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    print(f"  {len(out)} records → {dst}")


def main():
    sensor_files = sorted(LABELED_DIR.glob("TESTBED_*.jsonl"))

    print("=== Fixing data/labeled/ ===")
    for f in sensor_files:
        fix_file(f, f)          # overwrite in-place

    print("\n=== Fixing data/data/sensor_*.jsonl ===")
    for f in sorted(DATA_DIR.glob("sensor_*.jsonl")):
        fix_file(f, f)          # overwrite in-place

    # Verify uniqueness
    print("\n=== Verifying ===")
    for f in sorted(LABELED_DIR.glob("TESTBED_*.jsonl")):
        eids = [json.loads(l)["event_id"] for l in open(f) if l.strip()]
        unique = len(set(eids))
        print(f"  {f.name}: {len(eids)} records, {unique} unique event_ids",
              "✓" if len(eids) == unique else "✗ STILL HAS DUPES")


if __name__ == "__main__":
    main()
