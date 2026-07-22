"""
오프라인 검출기 테스트 (서버 없이 모델만 검증).

Process 1 로직을 직접 재현한다:
  1차 게이트: 센서(TinyAE) 또는 비전(Fire2DCNN) 중 하나라도 이상치면 트리거
  2차 확정 : 트리거 시 같은 event_id의 다른 모달리티까지 합쳐 FusionAE 멀티모달 판정
  확정 시  : disaster_type 태그 + ALERT 발동

시나리오 6가지 (트리거 모달 x 멀티모달 결과):
  sensor-alert   : 센서만 이상치, 멀티모달 확정 (ALERT)
  sensor-normal  : 센서만 이상치, 멀티모달 정상 (대조군, 오탐 억제)
  vision-alert   : 비전만 이상치, 멀티모달 확정 (ALERT)
  vision-normal  : 비전만 이상치, 멀티모달 정상 (대조군, 오탐 억제)
  both           : 센서+비전 동시 이상치, 멀티모달 확정
  normal         : 트리거 없음 (정상)

실행:
  cd iot_server && python tests/test_detector.py                    # 전체
  cd iot_server && python tests/test_detector.py --case sensor-alert
  cd iot_server && python tests/test_detector.py --case sensor-normal
  cd iot_server && python tests/test_detector.py --case vision-alert
  cd iot_server && python tests/test_detector.py --case vision-normal
  cd iot_server && python tests/test_detector.py --case both
  cd iot_server && python tests/test_detector.py --case normal
  cd iot_server && python tests/test_detector.py --scan 600         # 스캔 후보 수 조정
"""

import argparse
import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore")

# iot_server 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextmanager
def _quiet_stderr():
    """torch가 NumPy 1.x로 빌드돼 import 시 C 레벨에서 stderr로 쏟는 무해한
    NumPy 2.x 호환 경고를 파일 디스크립터 단에서 억제 (warnings 필터로는 안 잡힘)."""
    saved   = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


with _quiet_stderr():
    from detector import SensorDetector, VisionDetector, MultimodalAEDetector

REPO         = Path(__file__).resolve().parent.parent.parent
VISION_JSONL = REPO / "data" / "data" / "vision_labeled.jsonl"
LABELED_DIR  = REPO / "data" / "labeled"

# 버킷 키 -> 표시 라벨 (출력 순서 = 아래 dict 순서)
BUCKETS = {
    "sensor_alert":  "[1-1] 센서만 이상치, 멀티모달 확정 (ALERT)",
    "sensor_normal": "[대조군] 센서만 이상치, 멀티모달 정상 (오탐 억제)",
    "vision_alert":  "[1-2] 비전만 이상치, 멀티모달 확정 (ALERT)",
    "vision_normal": "[대조군] 비전만 이상치, 멀티모달 정상 (오탐 억제)",
    "both":          "센서+비전 동시 이상치, 멀티모달 확정",
    "normal":        "정상 (트리거 없음)",
}

# --case 선택값 -> 내부 버킷 키
CASE_MAP = {
    "sensor-alert":  "sensor_alert",
    "sensor-normal": "sensor_normal",
    "vision-alert":  "vision_alert",
    "vision-normal": "vision_normal",
    "both":          "both",
    "normal":        "normal",
}


def _load_maps():
    """event_id -> (센서 레코드, 비전 레코드) 매핑 + 후보 event_id 목록."""
    vision_map, fire_vids, normal_vids = {}, [], []
    with open(VISION_JSONL) as f:
        for line in f:
            r = json.loads(line)
            eid = r["event_id"]
            vision_map[eid] = r
            (fire_vids if r["metadata"]["anomaly_flag"] else normal_vids).append(eid)

    sensor_map, sensor_anom_vids = {}, []
    for p in sorted(LABELED_DIR.glob("TESTBED_*_labeled.jsonl")):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                eid = r.get("event_id")
                if not eid:
                    continue
                if r["metadata"].get("anomaly_flag"):
                    sensor_map[eid] = r            # 이상치 센서 레코드 우선
                    sensor_anom_vids.append(eid)
                else:
                    sensor_map.setdefault(eid, r)
    return vision_map, sensor_map, fire_vids, normal_vids, sensor_anom_vids


def _classify(sensor_anom, vision_anom, fusion_anom):
    if sensor_anom and vision_anom:
        return "both"
    if sensor_anom:
        return "sensor_alert" if fusion_anom else "sensor_normal"
    if vision_anom:
        return "vision_alert" if fusion_anom else "vision_normal"
    return "normal"


def _show(bucket_key, rec, sd):
    print(f"\n{BUCKETS[bucket_key]}")
    print(f"  event_id={rec['eid']}")
    blob = (rec["vrec"] or {}).get("data", {}).get("blob_uri")
    print(f"  [1차 게이트] sensor_anom={rec['sa']} (score={rec['ss']:.6f}, thr={sd.thr_anom:.6f})")
    print(f"              vision_anom={rec['va']} (score={rec['vs']}, blob={blob})  "
          f"트리거={rec['sa'] or rec['va']}")

    fz = rec["fz"]
    if fz is None:
        print("  [결과] 트리거 없음, Process 1 미발동 (정상)")
        return

    print(f"  [2차 융합] is_anomaly={fz['is_anomaly']} conf={fz['confidence']} "
          f"fusion_score={fz['fusion_score']:.6f} thr={fz['fusion_thr']:.6f}")
    if fz["is_anomaly"]:
        print(f"  [결과] ALERT 발동, tag={fz['disaster_type']}")
    else:
        print("  [결과] 단일 모달은 떴으나 멀티모달 미확정, 알림 보류 (오탐 억제)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["all"] + list(CASE_MAP), default="all",
                    help="실행할 시나리오 (기본 all)")
    ap.add_argument("--scan", type=int, default=600, help="스캔할 최대 후보 수")
    args = ap.parse_args()

    wanted = set(BUCKETS) if args.case == "all" else {CASE_MAP[args.case]}

    vision_map, sensor_map, fire_vids, normal_vids, sensor_anom_vids = _load_maps()
    print(f"화재 vision={len(fire_vids)} / 정상 vision={len(normal_vids)} / "
          f"센서이상치={len(sensor_anom_vids)}")

    print("\n검출기 로드 중 (TinyAE 임계값 산출에 수십 초 소요)...")
    with _quiet_stderr():
        sd = SensorDetector()
        vd = VisionDetector()
        md = MultimodalAEDetector(sd, vd)

    # 후보: 센서이상치 + 화재비전 + 정상 일부 (여러 조합을 만나도록 섞음)
    candidates, seen = [], set()
    for eid in sensor_anom_vids + fire_vids + normal_vids[:400]:
        if eid in seen or eid not in sensor_map:
            continue
        seen.add(eid)
        candidates.append(eid)
    candidates = candidates[: args.scan]

    # 원하는 시나리오 사례를 찾을 때까지 스캔 (트리거 시에만 FusionAE 실행)
    found = {}
    print(f"\n{len(candidates)}개 후보 스캔 중... (대상: {args.case})")
    for eid in candidates:
        srec = sensor_map[eid]
        vrec = vision_map.get(eid)
        sa, ss = sd.check(srec)
        va, vs = (vd.check(vrec) if vrec else (False, None))
        fz = md.fuse(srec, vrec) if (sa or va) else None
        key = _classify(sa, va, fz["is_anomaly"] if fz else False)
        if key not in found:
            found[key] = {
                "eid": eid, "srec": srec, "vrec": vrec,
                "sa": sa, "ss": ss, "va": va,
                "vs": (round(vs, 4) if vs is not None else None),
                "fz": fz,
            }
        if wanted <= set(found):
            break

    # 정해진 순서로 (선택된 것만) 출력
    for key in BUCKETS:
        if key not in wanted:
            continue
        if key in found:
            _show(key, found[key], sd)
        else:
            print(f"\n{BUCKETS[key]}\n  (해당 사례 없음)")

    print("\n[완료]")


if __name__ == "__main__":
    main()
