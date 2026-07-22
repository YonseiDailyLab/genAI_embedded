"""
ANDLAB push 클라이언트.
Process 1(이상치 즉시 알림) / Process 2-1(주기 전송) 모두 사용.
ANDLAB_ENDPOINT 환경변수로 엔드포인트 설정.
"""

import logging
import os

import httpx

log = logging.getLogger("sender")

ANDLAB_ENDPOINT = os.getenv(
    "ANDLAB_ENDPOINT",
    "http://165.132.192.52:7862/sensor/alert",
)
ANDLAB_TIMEOUT  = float(os.getenv("ANDLAB_TIMEOUT", "10"))


async def push_to_andlab(payload: dict, label: str = "") -> bool:
    """
    ANDLAB에 HTTP POST.
    본문은 sensor, image, text 세 모달리티를 포함한다.
    ANDLAB_ENDPOINT를 빈 문자열로 설정하면 전송을 비활성화한다.
    Returns True if sent successfully.
    """
    if not ANDLAB_ENDPOINT:
        tag = f"[{label}] " if label else ""
        log.debug(f"{tag}ANDLAB_ENDPOINT 미설정 -> 전송 스킵")
        return False

    try:
        async with httpx.AsyncClient(timeout=ANDLAB_TIMEOUT) as client:
            resp = await client.post(ANDLAB_ENDPOINT, json=payload)
            resp.raise_for_status()
            log.info(f"ANDLAB push OK [{label}] status={resp.status_code}")
            return True
    except Exception as e:
        log.warning(f"ANDLAB push 실패 [{label}]: {e}")
        return False
