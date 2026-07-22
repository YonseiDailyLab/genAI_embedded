"""
외부 요청 테스트 (Process 2-2) — 서버가 떠 있어야 한다.

stdlib(urllib)만 사용 -> 의존성/NumPy 경고 없이 어디서든 실행 가능.
다른 LAN(상대방)에서는 포트포워딩한 공인 IP:포트를 HOST/PORT로 주면 된다.

실행 (로컬):
  python tests/test_api.py
  python tests/test_api.py --host 127.0.0.1 --port 8002 --device 3C8427EE4928

실행 (외부 / 포트포워딩 45291 -> 8002):
  python tests/test_api.py --host <공인IP> --port 45291
  HOST=<공인IP> PORT=45291 python tests/test_api.py
"""

import argparse
import json
import os
import urllib.request


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _summarize_multimodal(d):
    det = d.get("detection", {})
    img = d.get("vision") or {}
    has_img = bool(img.get("data", {}).get("image_b64"))
    print(f"    is_anomaly  : {d.get('is_anomaly')}")
    print(f"    confidence  : {d.get('confidence')}")
    print(f"    disaster    : {d.get('disaster_type')}")
    print(f"    sensor_score: {det.get('sensor_score')}  thr={det.get('sensor_thr')}")
    print(f"    fusion_score: {det.get('fusion_score')}  thr={det.get('fusion_thr')}")
    print(f"    vision_score: {det.get('vision_score')}  flag={det.get('vision_flag')}")
    print(f"    image_b64 포함: {has_img}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", default=os.getenv("PORT", "8002"))
    ap.add_argument("--device", default="3C8427EE4928")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"대상 서버: {base}  device={args.device}\n")

    # 1) GET /health
    print("[1] GET /health")
    st, body = _get(f"{base}/health", args.timeout)
    print(f"    status={st}")
    print(f"    devices={body.get('devices')}")
    print(f"    stored_devices={body.get('stored_devices')}")

    # 2) POST /sensor/latest — 최근 홀드된 센서 데이터
    print("\n[2] POST /sensor/latest")
    st, body = _post(f"{base}/sensor/latest", {"device_id": args.device}, args.timeout)
    print(f"    status={st}")
    print(f"    event_id={body.get('event_id')} modality={body.get('modality')} "
          f"anomaly_flag={body.get('metadata', {}).get('anomaly_flag')}")

    # 3) POST /multimodal/latest — 최근 홀드된 멀티모달 이벤트 전체
    print("\n[3] POST /multimodal/latest")
    st, body = _post(f"{base}/multimodal/latest", {"device_id": args.device}, args.timeout)
    print(f"    status={st}")
    _summarize_multimodal(body)

    print("\n[완료]")


if __name__ == "__main__":
    main()
