"""
Sensor/Vision data loader.

FileReplayLoader: labeled JSONL를 읽어 현재 시각 기준으로 cyclic replay.
LiveLoader (stub): 실제 센서/카메라 연결 시 이 클래스를 구현.
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional


DATA_ROOT = Path(__file__).parent.parent / "data"
SENSOR_LABELED_DIR = DATA_ROOT / "labeled"
VISION_LABELED_FILE = DATA_ROOT / "data" / "vision_labeled.jsonl"
IMAGE_ROOT = DATA_ROOT / "data"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class FileReplayLoader:
    """
    Labeled JSONL 파일에서 현재 시각 대비 cyclic offset으로 최신 레코드를 반환.
    실제 센서 없이 ANDLAB 통합 테스트 가능.
    """

    def __init__(self):
        self._sensor: dict[str, list[dict]] = {}   # device_id -> sorted records
        self._vision_by_event: dict[str, dict] = {}  # event_id -> vision record
        self._dataset_start: int = 0
        self._dataset_span: int = 1
        self._load()

    def _load(self):
        for jsonl_path in sorted(SENSOR_LABELED_DIR.glob("TESTBED_*_labeled.jsonl")):
            records = _load_jsonl(jsonl_path)
            records.sort(key=lambda r: r["window"]["start"])
            device_id = records[0]["metadata"]["sensor"]["device_id"]
            self._sensor[device_id] = records

        # dataset 전체 시간 범위 (모든 디바이스 공통)
        all_starts = [r["window"]["start"] for recs in self._sensor.values() for r in recs]
        all_ends   = [r["window"]["end"]   for recs in self._sensor.values() for r in recs]
        self._dataset_start = min(all_starts)
        self._dataset_span  = max(all_ends) - self._dataset_start

        # vision: event_id 인덱스
        if VISION_LABELED_FILE.exists():
            for rec in _load_jsonl(VISION_LABELED_FILE):
                self._vision_by_event[rec["event_id"]] = rec

    def _current_dataset_time(self) -> int:
        """실제 시각을 데이터셋 범위 안에 cyclic mapping."""
        offset = int(time.time()) % self._dataset_span
        return self._dataset_start + offset

    def get_sensor_record(self, device_id: str) -> Optional[dict]:
        records = self._sensor.get(device_id)
        if not records:
            return None
        t = self._current_dataset_time()
        # t를 포함하는 window 탐색 (없으면 마지막 레코드)
        for rec in reversed(records):
            if rec["window"]["start"] <= t:
                return rec
        return records[-1]

    def get_vision_record(self, event_id: str) -> Optional[dict]:
        return self._vision_by_event.get(event_id)

    def list_devices(self) -> list[str]:
        return list(self._sensor.keys())


def load_image_b64(blob_uri: str) -> Optional[str]:
    """blob_uri (예: 'normal/1.jpg') -> base64 문자열."""
    img_path = IMAGE_ROOT / blob_uri
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
