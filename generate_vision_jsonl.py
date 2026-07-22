"""Build the shared data/data/ directory for cross-modal distribution.

Steps:
  1. Merge all labeled sensor JSONL files → data/data/sensor_labeled.jsonl
  2. Extract one video frame per event_id  → data/data/{정상|화재}/…/*.jpg
  3. Write vision JSONL                    → data/data/vision_labeled.jsonl
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
from tqdm import tqdm

LABELED_DIR    = Path("data/labeled")
VISION_DIR     = Path("data/vision")
OUT_DIR        = Path("data/data")
OUT_SENSOR     = OUT_DIR / "sensor_labeled.jsonl"
OUT_VISION     = OUT_DIR / "vision_labeled.jsonl"

LOCATION     = {"lat": 37.5663, "lon": 126.9432}
DEVICE_ID    = "cam1"
SCHEMA_VER   = "1.0"
IMG_W, IMG_H = 1920, 1080


# ── 0. Merge sensor JSONL files ───────────────────────────────────────────────

def copy_sensor_jsonl():
    """Copy each TESTBED_*.jsonl into data/data/ as sensor_{device_id}.jsonl."""
    sensor_files = sorted(LABELED_DIR.glob("TESTBED_*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in tqdm(sensor_files, desc="Copying sensor JSONL", unit="file"):
        dst = OUT_DIR / f"sensor_{f.stem}.jsonl"
        dst.write_text(f.read_text())
        count = sum(1 for l in dst.open() if l.strip())
        print(f"  {count} records → {dst.name}")


# ── 1. Load unique events (fire flag takes priority across devices) ────────────

def _event_id(window_start: int) -> str:
    bucket = (window_start // 60) * 60
    dt = datetime.fromtimestamp(bucket, tz=timezone.utc)
    return f"evt_{dt.strftime('%Y%m%d')}_{bucket}"


def load_events() -> dict:
    """One entry per unique window_start across all devices.
    anomaly_flag=True wins if any device flagged that window as anomaly.
    """
    events: dict = {}
    for f in sorted(LABELED_DIR.glob("TESTBED_*.jsonl")):
        for line in open(f):
            r     = json.loads(line)
            start = r["window"]["start"]
            flag  = r["metadata"]["anomaly_flag"]
            eid   = _event_id(start)
            if eid not in events or flag:
                events[eid] = {
                    "start":        start,
                    "end":          r["window"]["end"],
                    "anomaly_flag": flag,
                    "disaster_type": r["metadata"]["disaster_type"],
                }
    return events


# ── 2. Assign (video, frame_idx) to each event ────────────────────────────────

def assign_frames(events: dict, normal_videos: list, fire_videos: list) -> dict:
    """Return {event_id: (video_path, frame_idx, category)} for all events."""

    normal_eids = sorted(
        [eid for eid, v in events.items() if not v["anomaly_flag"]],
        key=lambda e: events[e]["start"],
    )
    fire_eids = sorted(
        [eid for eid, v in events.items() if v["anomaly_flag"]],
        key=lambda e: events[e]["start"],
    )

    assignments: dict = {}

    # Normal: distribute evenly across videos, space frames evenly within each video
    n_vids = len(normal_videos)
    # Group event indices by which video they'll use (round-robin on video)
    video_to_eids: dict = {}
    for i, eid in enumerate(normal_eids):
        vid = normal_videos[i % n_vids]
        video_to_eids.setdefault(vid, []).append(eid)

    for vid, eids in video_to_eids.items():
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n = len(eids)
        for j, eid in enumerate(eids):
            # space frames evenly; avoid last frame (may be incomplete)
            idx = math.floor(j * (total - 1) / max(n - 1, 1)) if n > 1 else total // 2
            idx = min(max(idx, 0), total - 1)
            assignments[eid] = (vid, idx, "정상")

    # Fire: one middle frame per fire video (cycle if more events than videos)
    for i, eid in enumerate(fire_eids):
        vid = fire_videos[i % len(fire_videos)]
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assignments[eid] = (vid, total // 2, "화재")

    return assignments


# ── 3. Extract frame and save JPEG ────────────────────────────────────────────

def extract_frame(video_path: Path, frame_idx: int, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return True
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(out_path), frame)
    return ret


# ── 4. Build JSONL record ─────────────────────────────────────────────────────

def make_record(event_id: str, ev: dict, blob_uri: str) -> dict:
    start = ev["start"]
    end   = ev["end"]
    return {
        "schema_version": SCHEMA_VER,
        "id":             f"img_{DEVICE_ID}:{start}-{end}",
        "event_id":       event_id,
        "modality":       "vision",
        "window": {
            "start": start,
            "end":   end,
            "size":  end - start,
        },
        "metadata": {
            "feature_dim": [IMG_H, IMG_W, 3],
            "location":    LOCATION,
            "anomaly_flag": ev["anomaly_flag"],
            "disaster_type": ev["disaster_type"],
            "sensor": {
                "device_id":   DEVICE_ID,
                "sampling_hz": 1 / 60,
                "units":       {},
            },
            "updated_at": end - 1,
        },
        "data": {
            "blob_uri": blob_uri,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Step 0: merge sensor data ─────────────────────────────────────────
    print("\n[1/3] Copying sensor JSONL (per device)...")
    copy_sensor_jsonl()

    # ── Step 1: load unique events ────────────────────────────────────────
    print("\n[2/3] Extracting vision frames...")
    events = load_events()
    n_normal = sum(1 for v in events.values() if not v["anomaly_flag"])
    n_fire   = sum(1 for v in events.values() if     v["anomaly_flag"])
    print(f"  {n_normal} normal events, {n_fire} fire events")

    # exclude macOS metadata files (._filename)
    normal_videos = sorted(f for f in (VISION_DIR / "정상").glob("*.mp4") if not f.name.startswith("._"))
    fire_videos   = sorted(f for f in (VISION_DIR / "화재").glob("*.mp4") if not f.name.startswith("._"))
    print(f"  {len(normal_videos)} normal videos, {len(fire_videos)} fire videos")

    assignments = assign_frames(events, normal_videos, fire_videos)

    records = []
    ok = 0
    counters = {"정상": 1, "화재": 1}
    dir_name = {"정상": "normal", "화재": "fire"}
    items = sorted(assignments.items(), key=lambda x: events[x[0]]["start"])
    for eid, (vid, frame_idx, category) in tqdm(items, unit="frame"):
        n        = counters[category]
        out_path = OUT_DIR / dir_name[category] / f"{n}.jpg"
        blob_uri = f"{dir_name[category]}/{n}.jpg"
        counters[category] += 1
        if extract_frame(vid, frame_idx, out_path):
            ok += 1
            records.append(make_record(eid, events[eid], blob_uri))
        else:
            tqdm.write(f"  WARNING: skipped {vid.name} @ {frame_idx}")
    print(f"  extracted {ok}/{len(assignments)} frames")

    # ── Step 2: write vision JSONL ────────────────────────────────────────
    print("\n[3/3] Writing vision JSONL...")
    with open(OUT_VISION, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records)} records → {OUT_VISION}")

    print(f"\nDone. data/data/ contents:")
    for p in sorted(OUT_DIR.rglob("*"))[:10]:
        print(f"  {p.relative_to(OUT_DIR)}")
    print("  ...")


if __name__ == "__main__":
    main()
