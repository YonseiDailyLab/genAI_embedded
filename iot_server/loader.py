"""
Sensor/Vision data loader.

FileReplayLoader: labeled JSONL를 읽어 현재 시각 기준으로 cyclic replay.
LiveLoader (stub): 실제 센서/카메라 연결 시 이 클래스를 구현.
"""

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:  # paho-mqtt 미설치 시 FileReplayLoader만 사용 가능
    mqtt = None

log = logging.getLogger("loader")


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


class LiveLoader:
    """
    디바이스가 MQTT로 publish하는 센서 윈도우(TinyAE/sensor)를 구독해
    device_id별 최신 레코드를 메모리에 보관한다.

    FileReplayLoader와 동일한 인터페이스(get_sensor_record / get_vision_record /
    list_devices)를 제공하므로 파이프라인·엔드포인트에서 그대로 교체 가능하다.

    디바이스는 센서 전용(카메라 없음)이라 get_vision_record는 항상 None을 반환한다.

    환경변수:
      MQTT_HOST          (기본 192.168.0.13)  — 디바이스 config.h kMqttHost와 일치 (이 Jetson)
      MQTT_PORT          (기본 1883)
      MQTT_TOPIC_SENSOR  (기본 TinyAE/sensor)  — 디바이스 kMqttTopicSensor와 일치
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        topic: Optional[str] = None,
        client_id: str = "iot-server",
    ):
        if mqtt is None:
            raise RuntimeError("paho-mqtt 미설치 — `pip install paho-mqtt` 후 사용")

        self._host  = host  or os.getenv("MQTT_HOST", "192.168.0.13")
        self._port  = port  or int(os.getenv("MQTT_PORT", "1883"))
        self._topic = topic or os.getenv("MQTT_TOPIC_SENSOR", "TinyAE/sensor")

        self._lock    = threading.Lock()
        self._latest: dict[str, dict] = {}   # device_id -> 최신 센서 레코드

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        # 브로커가 아직 안 떠 있어도 서버 기동을 막지 않도록 async 연결 + 자동 재접속.
        self._client.connect_async(self._host, self._port, keepalive=60)
        self._client.loop_start()   # 백그라운드 네트워크 스레드
        log.info(f"LiveLoader: MQTT 구독 시작 {self._host}:{self._port} topic={self._topic}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(self._topic, qos=0)
            log.info(f"LiveLoader: 브로커 연결 OK -> subscribe {self._topic}")
        else:
            log.warning(f"LiveLoader: 브로커 연결 실패 (rc={reason_code})")

    def _on_message(self, client, userdata, msg):
        try:
            rec = json.loads(msg.payload.decode("utf-8", "ignore"))
        except Exception as e:
            log.warning(f"LiveLoader: payload 파싱 실패: {e}")
            return
        device_id = rec.get("metadata", {}).get("sensor", {}).get("device_id")
        if not device_id:
            log.debug("LiveLoader: device_id 없는 레코드 무시")
            return
        with self._lock:
            self._latest[device_id] = rec

    # -- FileReplayLoader 호환 인터페이스 --------------------------------------

    def get_sensor_record(self, device_id: str) -> Optional[dict]:
        with self._lock:
            return self._latest.get(device_id)

    def get_vision_record(self, event_id: str) -> Optional[dict]:
        return None   # 디바이스는 센서 전용 → 비전 레코드 없음

    def list_devices(self) -> list[str]:
        with self._lock:
            return list(self._latest.keys())

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def load_image_b64(blob_uri: str) -> Optional[str]:
    """blob_uri (예: 'normal/1.jpg') -> base64 문자열."""
    img_path = IMAGE_ROOT / blob_uri
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
