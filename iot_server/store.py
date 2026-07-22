"""
로컬 이벤트 저장소.
Process 2-1에서 hold, Process 2-2(on-demand API)에서 serve.
"""

import threading
from collections import deque


class EventStore:
    """
    device_id별 최근 이벤트를 maxlen개 보관.
    스레드 안전 (asyncio + 백그라운드 스레드 혼용 대응).
    """

    def __init__(self, maxlen: int = 200):
        self._lock    = threading.Lock()
        self._history: dict[str, deque] = {}   # device_id -> deque[event]
        self._latest:  dict[str, dict]  = {}   # device_id -> most recent event
        self._maxlen  = maxlen

    def put(self, device_id: str, event: dict) -> None:
        with self._lock:
            if device_id not in self._history:
                self._history[device_id] = deque(maxlen=self._maxlen)
            self._history[device_id].append(event)
            self._latest[device_id] = event

    def get_latest(self, device_id: str) -> dict | None:
        with self._lock:
            return self._latest.get(device_id)

    def get_latest_any(self) -> dict | None:
        """device_id 무관, 가장 최근 이벤트 반환."""
        with self._lock:
            if not self._latest:
                return None
            return max(self._latest.values(), key=lambda e: e.get("timestamp", 0))

    def get_history(self, device_id: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(device_id, []))

    def list_devices(self) -> list[str]:
        with self._lock:
            return list(self._latest.keys())
