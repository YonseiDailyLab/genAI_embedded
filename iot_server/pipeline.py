"""
메인 파이프라인 — Process 1 + Process 2 (독립 루프).

Process 1 (이벤트 기반, 짧은 주기 PROC1_INTERVAL):
  단일 모달(센서 TinyAE 또는 비전 Fire2DCNN) 중 하나라도 이상치 -> 트리거.
  트리거 시 같은 event_id의 다른 모달리티까지 합쳐 FusionAE 멀티모달 확정.
  멀티모달도 이상치면 -> disaster_type 태그 + ANDLAB 즉시 push.
  같은 event_id 중복 알림은 스킵.

Process 2 (주기 보고, POLL_INTERVAL):
  2-1. 트리거 무관하게 멀티모달 판정 -> 로컬 store hold + ANDLAB push.
  2-2. 외부 요청 시 store에서 최신 데이터 반환 (-> main.py FastAPI 담당).

두 프로세스는 asyncio.gather로 병렬 실행된다.
"""

import asyncio
import base64
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detector import MultimodalAEDetector, MultimodalFuser
from gas_sensor_vlm import generate_report
from loader   import FileReplayLoader, IMAGE_ROOT
from sender   import push_to_andlab
from store    import EventStore

log = logging.getLogger("pipeline")

# Process 1: 이벤트 기반 — 단일 모달 이상치를 빠르게 감지하기 위한 짧은 폴링 주기
PROC1_INTERVAL = int(os.getenv("PROC1_INTERVAL", "5"))    # 초
# Process 2: 주기 보고 — 트리거 무관 hold + push
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL", "60"))    # 초
VLM_BASE_URL   = os.getenv("VLM_BASE_URL", "http://127.0.0.1:9000/v1")
VLM_MODEL      = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_TIMEOUT    = float(os.getenv("VLM_TIMEOUT", "120"))
VLM_RETRIES    = int(os.getenv("VLM_RETRIES", "2"))


def _load_image_b64(blob_uri: str) -> str | None:
    path = IMAGE_ROOT / blob_uri
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_sensor_payload(sensor_record: dict, fusion: dict) -> dict:
    """Build the sensor JSON shared by Pull and ANDLAB Push."""
    return {
        **sensor_record,
        "metadata": {
            **sensor_record.get("metadata", {}),
            "anomaly_flag": bool(fusion["is_anomaly"]),
            "disaster_type": fusion["disaster_type"],
        },
    }


def _build_event_payload(
    device_id: str,
    sensor_record: dict,
    vision_record: dict | None,
    fusion: dict,
) -> dict:
    """로컬 store용 멀티모달 페이로드 구성 (센서 + 이미지 결합)."""
    sensor_payload = _build_sensor_payload(sensor_record, fusion)
    vision_payload = None
    if vision_record:
        blob_uri = vision_record.get("data", {}).get("blob_uri", "")
        vision_payload = {
            **vision_record,
            "data": {
                **vision_record.get("data", {}),
                "image_b64": _load_image_b64(blob_uri),
            },
        }

    return {
        "timestamp":    int(time.time()),
        "device_id":    device_id,
        "event_id":     sensor_payload.get("event_id"),
        "is_anomaly":   fusion["is_anomaly"],
        "confidence":   fusion["confidence"],
        "disaster_type": fusion["disaster_type"],
        "detection": {
            "sensor_score":  fusion["sensor_score"],
            "sensor_thr":    fusion["sensor_thr"],
            "vision_score":  fusion.get("vision_score"),
            "vision_flag":   fusion.get("vision_flag"),
            "fusion_score":  fusion.get("fusion_score"),
            "fusion_thr":    fusion.get("fusion_thr"),
        },
        "sensor": sensor_payload,
        "vision": vision_payload,
    }


def _fetch_event(loader: FileReplayLoader, device_id: str):
    """(sensor_record, event_id, vision_record) — 같은 event_id로 두 모달리티 결합."""
    sensor_record = loader.get_sensor_record(device_id)
    if sensor_record is None:
        return None, None, None
    event_id      = sensor_record.get("event_id")
    vision_record = loader.get_vision_record(event_id) if event_id else None
    return sensor_record, event_id, vision_record


async def _push_event(
    device_id: str,
    event: dict,
    label: str,
    store: EventStore,
) -> bool:
    """Add Qwen text and Push the sensor, image, and text modalities."""
    previous = store.get_latest(device_id)
    text = (
        previous.get("text")
        if previous and previous.get("event_id") == event.get("event_id")
        else None
    )
    image = event.get("vision")
    image_b64 = (image or {}).get("data", {}).get("image_b64")

    # ponytail: inline inference blocks this device loop; use a queue if latency affects polling.
    if not text and image_b64:
        text = await generate_report(
            event["sensor"],
            image_b64,
            event["sensor"].get("metadata", {})
                .get("disaster_type", {})
                .get("main_tag", "normal"),
            base_url=VLM_BASE_URL,
            model=VLM_MODEL,
            timeout=VLM_TIMEOUT,
            retries=VLM_RETRIES,
        )

    event["text"] = text
    store.put(device_id, event)
    if not image_b64 or not text:
        log.warning(
            f"[{label}:{device_id}] ANDLAB 전송 스킵: "
            f"image={'ok' if image_b64 else 'missing'}, text={'ok' if text else 'missing'}"
        )
        return False

    return await push_to_andlab({
        "sensor": event["sensor"],
        "image": image,
        "text": text,
    }, label=label)


# -- 진입점: 두 프로세스 병렬 실행 ---------------------------------------------

async def run_pipeline(
    loader: FileReplayLoader,
    fuser:  MultimodalAEDetector | MultimodalFuser,
    store:  EventStore,
) -> None:
    """FastAPI lifespan에서 백그라운드 태스크로 실행. Process 1 + 2 동시 구동."""
    log.info(
        f"파이프라인 시작 | Process1={PROC1_INTERVAL}s(이벤트 기반) "
        f"Process2={POLL_INTERVAL}s(주기)"
    )
    await asyncio.gather(
        _process1_loop(loader, fuser, store),
        _process2_loop(loader, fuser, store),
    )


# -- Process 1: 이벤트 기반 (단일 모달 OR 게이트 -> 멀티모달 확정) ---------------

async def _process1_loop(
    loader: FileReplayLoader,
    fuser:  MultimodalAEDetector | MultimodalFuser,
    store:  EventStore,
) -> None:
    last_alert: dict[str, str] = {}   # device_id -> 마지막 알림 event_id (중복 방지)

    while True:
        await asyncio.sleep(PROC1_INTERVAL)
        for device_id in loader.list_devices():
            try:
                await _process1_device(device_id, loader, fuser, store, last_alert)
            except Exception as e:
                log.error(f"[P1:{device_id}] 처리 오류: {e}", exc_info=True)


async def _process1_device(
    device_id:  str,
    loader:     FileReplayLoader,
    fuser:      MultimodalAEDetector | MultimodalFuser,
    store:      EventStore,
    last_alert: dict[str, str],
) -> None:
    sensor_record, event_id, vision_record = _fetch_event(loader, device_id)
    if sensor_record is None:
        return

    # 1차: 단일 모달 트리거 — 센서 또는 비전 중 하나라도 이상치
    sensor_anom, _ = fuser.sensor.check(sensor_record)
    vision_anom    = False
    if vision_record is not None:
        vision_anom, _ = fuser.vision.check(vision_record)

    if not (sensor_anom or vision_anom):
        return   # 트리거 없음 -> Process 1 종료

    # 같은 이벤트 중복 알림 방지
    if last_alert.get(device_id) == event_id:
        return

    # 2차: 다른 모달리티 결합 -> FusionAE 멀티모달 확정
    fusion = fuser.fuse(sensor_record, vision_record)
    if not fusion["is_anomaly"]:
        return   # 단일 모달은 떴으나 멀티모달에서 미확정 -> 알림 보류

    last_alert[device_id] = event_id
    event = _build_event_payload(device_id, sensor_record, vision_record, fusion)
    tag   = fusion["disaster_type"].get("main_tag", "anomaly")
    trig  = "sensor" if sensor_anom else "vision"
    log.warning(
        f"[P1:{device_id}] 멀티모달 이상치 확정! trigger={trig} type={tag} "
        f"fusion_score={fusion.get('fusion_score')}"
    )
    await _push_event(device_id, event, f"ALERT:{tag}", store)


# -- Process 2: 주기 보고 (트리거 무관, 항상 hold + push) -----------------------

async def _process2_loop(
    loader: FileReplayLoader,
    fuser:  MultimodalAEDetector | MultimodalFuser,
    store:  EventStore,
) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        t0 = time.time()
        for device_id in loader.list_devices():
            try:
                await _process2_device(device_id, loader, fuser, store)
            except Exception as e:
                log.error(f"[P2:{device_id}] 처리 오류: {e}", exc_info=True)
        log.debug(f"[P2] 주기 완료 ({time.time()-t0:.1f}s)")


async def _process2_device(
    device_id: str,
    loader:    FileReplayLoader,
    fuser:     MultimodalAEDetector | MultimodalFuser,
    store:     EventStore,
) -> None:
    sensor_record, _, vision_record = _fetch_event(loader, device_id)
    if sensor_record is None:
        return

    # 트리거 무관하게 멀티모달 판정 -> hold + push
    fusion = fuser.fuse(sensor_record, vision_record)
    event  = _build_event_payload(device_id, sensor_record, vision_record, fusion)
    await _push_event(device_id, event, "PERIODIC", store)
