"""ANDLAB Push contract checks without network access."""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline


class _Loader:
    def __init__(self, sensor_record):
        self.sensor_record = sensor_record

    def get_sensor_record(self, device_id):
        return self.sensor_record

    def get_vision_record(self, event_id):
        return {
            "event_id": event_id,
            "metadata": {"disaster_type": {"main_tag": "normal"}},
            "data": {"blob_uri": "normal/1.jpg"},
        }


class _Detector:
    def check(self, record):
        return True, 1.0


class _Fuser:
    sensor = _Detector()
    vision = _Detector()

    def __init__(self, is_anomaly):
        self.is_anomaly = is_anomaly

    def fuse(self, sensor_record, vision_record):
        return {
            "is_anomaly": self.is_anomaly,
            "confidence": "sensor_only" if self.is_anomaly else "normal",
            "disaster_type": {
                "main_tag": "fire" if self.is_anomaly else "normal",
                "sub_tag": [],
            },
            "sensor_score": 1.0,
            "sensor_thr": 0.5,
            "vision_score": None,
            "vision_flag": None,
        }


class _Store:
    def get_latest(self, device_id):
        return getattr(self, "event", None)

    def put(self, device_id, event):
        self.event = event


class PushContractTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sensor_record = {
            "event_id": "test-001",
            "metadata": {
                "anomaly_flag": False,
                "disaster_type": {"main_tag": "unknown"},
            },
            "data": {"Temperature": [70], "CO2": [900]},
        }

    async def test_periodic_push_reports_normal_and_anomaly_sensor_json(self):
        for is_anomaly in (False, True):
            store = _Store()
            push = AsyncMock(return_value=True)
            report = AsyncMock(return_value="The image shows normal sensor readings. However, no emergency is present.")
            with (
                patch.object(pipeline, "push_to_andlab", push),
                patch.object(pipeline, "generate_report", report),
                patch.object(pipeline, "_load_image_b64", return_value="jpeg-base64"),
            ):
                await pipeline._process2_device(
                    "device-1", _Loader(self.sensor_record), _Fuser(is_anomaly), store
                )

            payload = push.await_args.args[0]
            self.assertEqual(set(payload), {"sensor", "image", "text"})
            self.assertEqual(payload["sensor"], store.event["sensor"])
            self.assertEqual(payload["sensor"]["metadata"]["anomaly_flag"], is_anomaly)
            self.assertEqual(payload["sensor"]["data"], self.sensor_record["data"])
            self.assertEqual(payload["image"]["data"]["image_b64"], "jpeg-base64")
            self.assertEqual(payload["text"], store.event["text"])
            self.assertEqual(push.await_args.kwargs["label"], "PERIODIC")

        self.assertFalse(self.sensor_record["metadata"]["anomaly_flag"])

    async def test_immediate_alert_push_uses_sensor_json(self):
        store = _Store()
        last_alert = {}
        push = AsyncMock(return_value=True)
        report = AsyncMock(return_value="The image shows fire sensor readings. Furthermore, a fire emergency is confirmed.")
        with (
            patch.object(pipeline, "push_to_andlab", push),
            patch.object(pipeline, "generate_report", report),
            patch.object(pipeline, "_load_image_b64", return_value="jpeg-base64"),
        ):
            await pipeline._process1_device(
                "device-1",
                _Loader(self.sensor_record),
                _Fuser(True),
                store,
                last_alert,
            )

        payload = push.await_args.args[0]
        self.assertEqual(set(payload), {"sensor", "image", "text"})
        self.assertTrue(payload["sensor"]["metadata"]["anomaly_flag"])
        self.assertEqual(payload["text"], store.event["text"])
        self.assertEqual(push.await_args.kwargs["label"], "ALERT:fire")

if __name__ == "__main__":
    unittest.main()
