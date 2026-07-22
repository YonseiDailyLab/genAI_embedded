"""
Daily Lab IoT API Server
========================
Process 1/2 파이프라인을 백그라운드로 실행하면서,
외부 요청(Process 2-2)은 FastAPI 엔드포인트로 서빙.

환경변수:
  ANDLAB_ENDPOINT   : ANDLAB push URL (기본값 http://165.132.192.52:7862/sensor/alert)
  VLM_BASE_URL      : Qwen vLLM OpenAI API URL (기본값 http://127.0.0.1:9000/v1)
  VLM_MODEL         : Qwen 모델 이름 (기본값 Qwen/Qwen2.5-VL-3B-Instruct)
  VIS_CKPT          : VisionDetector MLPEmbedder checkpoint 경로 (없으면 labeled fallback)
  VIS_FIRE_THR      : MLP sigmoid 임계값 (기본 0.5)
  PORT              : 서버 포트 (기본 8002)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector  import MultimodalAEDetector, MultimodalFuser, SensorDetector, VisionDetector
from loader    import FileReplayLoader, load_image_b64
from pipeline  import run_pipeline
from store     import EventStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")


# -- 전역 컴포넌트 --------------------------------------------------------------

loader: FileReplayLoader | None                        = None
store:  EventStore | None                              = None
fuser:  MultimodalAEDetector | MultimodalFuser | None  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, store, fuser

    log.info("컴포넌트 초기화 중...")

    loader = FileReplayLoader()

    # SensorDetector: TinyAE + Tukey IQR 임계값 산출 (수십 초 소요)
    log.info("TinyAE 임계값 산출 중 (최초 1회)...")
    sensor_det = SensorDetector()
    log.info(f"  thr_warn={sensor_det.thr_warn:.6f}  thr_anom={sensor_det.thr_anom:.6f}")

    # VisionDetector: vision/checkpoints/vision_best.pt 자동 로드 (없으면 labeled fallback)
    vision_det = VisionDetector()

    # FusionAE checkpoint 있으면 MultimodalAEDetector, 없으면 규칙 기반 fallback
    fuser = MultimodalAEDetector(sensor_det=sensor_det, vision_det=vision_det)
    if fuser.has_model:
        log.info("FusionAE 멀티모달 검출기 활성화")
    else:
        log.info("FusionAE checkpoint 없음 -> 규칙 기반 fuser 사용")
    store = EventStore(maxlen=200)

    # 백그라운드 파이프라인 시작
    task = asyncio.create_task(run_pipeline(loader, fuser, store))
    log.info("파이프라인 백그라운드 태스크 시작")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    log.info("서버 종료")


app = FastAPI(title="Daily Lab IoT API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- 요청 모델 -----------------------------------------------------------------

class SensorRequest(BaseModel):
    device_id: str
    window: int = 60


# -- 엔드포인트 ----------------------------------------------------------------

@app.get("/health")
def health():
    devices = loader.list_devices() if loader else []
    stored  = store.list_devices()  if store  else []
    return {
        "status":          "ok",
        "devices":         devices,
        "stored_devices":  stored,
    }


@app.post("/sensor/latest")
def sensor_latest(req: SensorRequest):
    """
    Process 2-2: 가장 최근 hold된 센서 데이터 반환.
    store에 데이터 없으면 loader에서 직접 반환 (서버 초기 기간 대응).
    """
    event = store.get_latest(req.device_id) if store else None

    if event:
        return event["sensor"]

    # store에 없으면 loader fallback
    if loader:
        record = loader.get_sensor_record(req.device_id)
        if record:
            return record

    raise HTTPException(404, f"device_id '{req.device_id}' not found")


@app.post("/multimodal/latest")
def multimodal_latest(req: SensorRequest):
    """
    Process 2-2: 가장 최근 hold된 멀티모달 이벤트 전체 반환.
    detection 결과(is_anomaly, confidence, disaster_type)도 포함.
    """
    event = store.get_latest(req.device_id) if store else None

    if event:
        return event

    # store에 없으면 loader fallback (detection 결과 없이)
    if loader:
        sensor_record = loader.get_sensor_record(req.device_id)
        if sensor_record:
            event_id     = sensor_record.get("event_id")
            vision_record = loader.get_vision_record(event_id) if event_id else None
            image_b64    = None
            if vision_record:
                blob_uri  = vision_record.get("data", {}).get("blob_uri", "")
                image_b64 = load_image_b64(blob_uri)
            return {
                "device_id":  req.device_id,
                "event_id":   event_id,
                "is_anomaly": None,
                "confidence": "pending",
                "sensor":     sensor_record,
                "vision": {
                    **vision_record,
                    "data": {**vision_record.get("data", {}), "image_b64": image_b64},
                } if vision_record else None,
            }

    raise HTTPException(404, f"device_id '{req.device_id}' not found")


# -- 실행 ----------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
