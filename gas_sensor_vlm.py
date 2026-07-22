#!/usr/bin/env python3
"""
sensor_TESTBED_*_labeled.jsonl + vision_labeled.jsonl
→ Qwen2.5-VL-3B-Instruct → sensor_image_reports.jsonl

Per sensor row: 2 English sentences, sensor-value-centered.
Describes how sensor readings are visually reflected in the matched image.
"""
import argparse
import asyncio
import json
import re
import time
import cv2
import httpx
from pathlib import Path
try:
    from openai import OpenAI, APIConnectionError
except ImportError:
    OpenAI = None

    class APIConnectionError(Exception):
        pass


PROJECT_ROOT  = Path(__file__).resolve().parent
DISASTER_DIR  = PROJECT_ROOT / "data" / "data"
VISION_JSONL  = DISASTER_DIR / "vision_labeled.jsonl"
SENSOR_JSONLS = sorted((PROJECT_ROOT / "data" / "labeled").glob("TESTBED_*_labeled.jsonl"))
HOST_ROOT     = PROJECT_ROOT
DOCKER_ROOT   = Path("/workspace")
RESIZE_DIR    = DISASTER_DIR / "_resized"
OUTPUT_JSONL  = PROJECT_ROOT / "sensor_image_reports.jsonl"

BASE_URL   = "http://127.0.0.1:9000/v1"
MODEL      = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_TOKENS = 220
MAX_EDGE   = 640

# MQ131_O3_ppb excluded — all INT32_MAX (broken sensor)
SENSOR_KEYS = [
    "Temperature", "Humidity", "CO2", "PM10", "PM2.5",
    "MP801_raw", "CO_SEN0564_raw", "MiCS2714_raw",
    "BME_Temp", "BME_Humidity", "ENS_AQI", "ENS_TVOC_ppb", "ENS_eCO2_ppm",
]

INT32_MAX = 2147483647


# ── helpers ───────────────────────────────────────────────────────────────────

def array_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None and v != INT32_MAX and v != -1]
    return round(sum(clean) / len(clean), 2) if clean else None


def fmt(v: float | None, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if v is not None else "n/a"


def to_docker_path(host_path: Path) -> Path:
    rel = host_path.resolve().relative_to(HOST_ROOT.resolve())
    return DOCKER_ROOT / rel


def resize_image(src: Path) -> Path:
    RESIZE_DIR.mkdir(exist_ok=True)
    rel = src.relative_to(DISASTER_DIR)
    out = RESIZE_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Cannot read image: {src}")
    h, w = img.shape[:2]
    if max(h, w) > MAX_EDGE:
        scale = MAX_EDGE / max(h, w)
        img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img)
    return out


# ── sensor processing ─────────────────────────────────────────────────────────

def compute_stats(data: dict) -> dict:
    """min, max, mean per sensor key."""
    stats = {}
    for k in SENSOR_KEYS:
        clean = [v for v in data.get(k, []) if v is not None and v != INT32_MAX and v != -1]
        if not clean:
            stats[k] = None
        else:
            stats[k] = {
                "min":  round(min(clean), 1),
                "max":  round(max(clean), 1),
                "mean": round(sum(clean) / len(clean), 1),
            }
    return stats


SENSOR_PRIORITY = {
    "fire":   ["PM10", "PM2.5", "Temperature", "CO2", "ENS_TVOC_ppb", "MiCS2714_raw"],
    "normal": ["Temperature", "CO2", "PM10", "PM2.5", "Humidity", "ENS_eCO2_ppm"],
}


def build_sensor_summary(stats: dict, units: dict, label: str, n: int = 4) -> str:
    """Return top-n most relevant sensors for the label as formatted lines."""
    lines = []
    for k in SENSOR_PRIORITY.get(label, SENSOR_KEYS):
        s = stats.get(k)
        if s is None:
            continue
        unit = units.get(k, "")
        unit_str = f" {unit}" if unit else ""
        if s["min"] == s["max"]:
            lines.append(f"  {k}: {s['mean']}{unit_str}")
        else:
            lines.append(f"  {k}: {s['min']}-{s['max']}{unit_str}")
        if len(lines) == n:
            break
    return "\n".join(lines)


# ── prompt & validation ───────────────────────────────────────────────────────

def build_prompt(sensor_summary: str, label: str) -> str:
    if label == "fire":
        s1 = (
            "S1: Start with 'The image shows' and describe the visible fire evidence "
            "(flames, smoke, burned area, etc.), then cite 2 sensor readings with min-max ranges and units."
        )
        s2 = "S2: State that the image and sensor readings confirm a fire-related emergency."
    else:
        s1 = (
            "S1: Start with 'The image shows' and describe the visible scene "
            "(street, room, outdoor area, etc.) and conditions, "
            "then cite 2 sensor readings with min-max ranges and units."
        )
        s2 = "S2: State that the image and sensor readings confirm no fire or emergency."

    return (
        "You are an environmental sensor analyst.\n"
        "Write exactly 2 sentences. No bullets or numbering.\n"
        "Use only the values below — do not invent numbers.\n\n"
        f"{s1}\n"
        f"{s2}\n\n"
        "Sensor readings:\n"
        f"{sensor_summary}\n"
    )


def normalize_two_sentences(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [re.sub(r"^\*{0,2}\s*(sentence|line)\s*\d+\s*:\s*\*{0,2}\s*", "", ln,
                    flags=re.IGNORECASE).strip() for ln in lines]
    lines = [re.sub(r"^\d+\s*[\).]\s*", "", ln).strip() for ln in lines]
    lines = [ln for ln in lines if ln]
    return " ".join(lines) if lines else text.strip()


def call_vlm(client: OpenAI, model: str, prompt: str, img_url: str,
             retries: int) -> str | None:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": img_url}},
    ]}]
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=MAX_TOKENS, temperature=0.25,
            )
        except APIConnectionError as e:
            print(f"  [conn error] attempt {attempt+1}: {e}")
            time.sleep(5)
            continue
        text = normalize_two_sentences((resp.choices[0].message.content or "").strip())
        if text:
            return text
    return None


async def generate_report(
    sensor_record: dict,
    image_b64: str,
    label: str,
    base_url: str = BASE_URL,
    model: str = MODEL,
    timeout: float = 120.0,
    retries: int = 2,
) -> str | None:
    """Generate one grounded report through a vLLM OpenAI-compatible endpoint."""
    stats = compute_stats(sensor_record.get("data", {}))
    units = sensor_record.get("metadata", {}).get("sensor", {}).get("units", {})
    sensor_summary = build_sensor_summary(stats, units, label)
    if not sensor_summary or not image_b64:
        return None

    prompt = build_prompt(sensor_summary, label)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]}]
    url = f"{base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.post(url, json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.25,
                })
                response.raise_for_status()
                text = normalize_two_sentences(
                    (response.json()["choices"][0]["message"]["content"] or "").strip()
                )
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                print(f"  [vlm error] attempt {attempt + 1}: {exc}")
                if attempt < retries:
                    await asyncio.sleep(1)
                continue

            if text:
                return " ".join(text.split())
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def load_vision_index(jsonl_path: Path) -> dict:
    """event_id → {blob_uri, window_start, label}"""
    idx = {}
    skipped = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] vision_labeled.jsonl line {i} skipped (parse error)")
                skipped += 1
                continue
            idx[d["event_id"]] = {
                "blob_uri":     d["data"]["blob_uri"],
                "window_start": int(d["window"]["start"]),
                "label":        d["metadata"]["disaster_type"]["main_tag"],
            }
    if skipped:
        print(f"  [warn] {skipped} broken lines skipped in {jsonl_path.name}")
    return idx


def iter_sensor_rows(jsonl_path: Path, vision_idx: dict):
    """
    Yield best-matching sensor row per (device_id, event_id) pair.
    When multiple rows share the same event_id, pick the one whose
    window start is closest to the vision record's window start.
    """
    # Group rows by event_id
    groups: dict[str, list] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = d["event_id"]
            if eid not in vision_idx:
                continue
            groups.setdefault(eid, []).append(d)

    for eid, rows in groups.items():
        vis_start = vision_idx[eid]["window_start"]
        best = min(rows, key=lambda r: abs(int(r["window"]["start"]) - vis_start))
        yield best


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-jsonl", type=Path, action="append", default=None,
                        help="Sensor JSONL path(s). Defaults to all sensor_TESTBED_*_labeled.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to process per sensor file (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt without calling VLM")
    parser.add_argument("--retries", type=int, default=2,
                        help="VLM retry count per row")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    sensor_files = args.sensor_jsonl or SENSOR_JSONLS
    client = None
    if not args.dry_run:
        if OpenAI is None:
            parser.error("openai package is required for batch generation")
        client = OpenAI(base_url=args.base_url, api_key="dummy", timeout=args.timeout)
        print("[server] waiting for vLLM server...", flush=True)
        for i in range(60):
            try:
                client.models.list()
                print("[server] ready", flush=True)
                break
            except APIConnectionError:
                print(f"[server] not ready yet ({i+1}/60), retrying in 5s...", flush=True)
                time.sleep(5)
        else:
            print("[server] ERROR: server did not respond after 5 minutes. Exiting.")
            return

    # Load already-processed (event_id, device_id) to support resume
    done: set[tuple[str, str]] = set()
    if not args.dry_run and args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["event_id"], rec["device_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done)} rows already done")

    vision_idx = load_vision_index(VISION_JSONL)
    print(f"[vision] {len(vision_idx)} events loaded")

    out_f = None if args.dry_run else args.out.open("a", encoding="utf-8")
    total = 0

    try:
        for sensor_path in sensor_files:
            print(f"\n[sensor] {sensor_path.name}")
            processed = 0

            for row in iter_sensor_rows(sensor_path, vision_idx):
                event_id  = row["event_id"]
                device_id = row["metadata"]["sensor"]["device_id"]
                units     = row["metadata"]["sensor"]["units"]
                label     = vision_idx[event_id]["label"]
                blob_uri  = vision_idx[event_id]["blob_uri"]

                if (event_id, device_id) in done:
                    continue

                stats   = compute_stats(row["data"])
                summary = build_sensor_summary(stats, units, label)
                prompt  = build_prompt(summary, label)

                img_host = DISASTER_DIR / blob_uri
                if not img_host.exists():
                    print(f"  [skip] image not found: {img_host}")
                    continue

                if args.dry_run:
                    print(f"\n=== {event_id} | {device_id} | {label} | {blob_uri} ===")
                    print(prompt)
                    processed += 1
                    if args.limit and processed >= args.limit:
                        break
                    continue

                try:
                    img_resized = resize_image(img_host)
                except RuntimeError as e:
                    print(f"  [skip] {e}")
                    continue

                img_url = f"file://{to_docker_path(img_resized)}"
                report  = call_vlm(
                    client, args.model, prompt, img_url, args.retries,
                )

                if report is None:
                    print(f"  [skip] VLM failed: {event_id} | {device_id}")
                    continue

                record = {
                    "event_id":  event_id,
                    "device_id": device_id,
                    "label":     label,
                    "image":     blob_uri,
                    "sensor_stats": {k: v for k, v in stats.items() if v is not None},
                    "report":    " ".join(report.split()),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                processed += 1
                total += 1
                print(f"  [{label:6s}] {event_id} | {device_id} → {report!r}")

                if args.limit and processed >= args.limit:
                    break

    finally:
        if out_f:
            out_f.close()

    if args.dry_run:
        print("\n[dry-run] done — no VLM calls made")
    else:
        print(f"\n[done] {total} rows written to {args.out}")


if __name__ == "__main__":
    main()
