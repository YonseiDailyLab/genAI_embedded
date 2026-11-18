#include "tinyae/app/pipeline.h"

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"

#include <Arduino.h>
#include <SD.h>
#include <WiFi.h>
#include <PubSubClient.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <time.h>

#include "tinyae/app/config.h"
#include "tinyae/data/sensors.h"
#include "tinyae/ml/autoencoder.h"

namespace {

static const char* TAG = "TinyAE_Pipe";

// AE input = 15 features × 6 statistics = 90.
constexpr uint32_t kSampleDim     = tinyae::ml::kAutoencoderInputDim;  // 90
constexpr uint32_t kFeatureDim    = tinyae::data::kFeatureCount;        // 15
constexpr uint32_t kStatsWindowSz = 300;  // rolling history buffer size (seconds)
constexpr uint32_t kTrainStride   = 150;  // push new training sample every N seconds
constexpr uint32_t kInferInterval = 60;   // run inference every N seconds (real-time)
static_assert(kFeatureDim * tinyae::data::kStatsPerFeature == kSampleDim,
              "kFeatureDim * kStatsPerFeature must equal kSampleDim");
static_assert(kTrainStride <= kStatsWindowSz, "stride must be <= window");
static_assert(kInferInterval <= kStatsWindowSz, "infer interval must be <= window");

// 1 day of training samples at stride 150 s: 86400/150 = 576 windows (576×60×4 = 138 KB PSRAM)
constexpr uint32_t kRingCapacity = 576;
constexpr uint32_t kTrainBatch   = 16;
constexpr uint32_t kMqttWindowSz = 60;

// ---------------------------------------------------------------------------
// MQTT window payload (raw sensor values for one 60-sample window).
// Allocated from PSRAM by comm_task, freed by mqtt_task after publish.
// ---------------------------------------------------------------------------
struct MqttWindowData {
  time_t   window_start;
  uint32_t event_seq;
  float    temperature[kMqttWindowSz];
  float    humidity[kMqttWindowSz];
  int      co2[kMqttWindowSz];
  int      pm10[kMqttWindowSz];
  int      pm25[kMqttWindowSz];
  int      mp801_raw[kMqttWindowSz];
  int      mq131_o3_ppb[kMqttWindowSz];
  int      co_sen0564_raw[kMqttWindowSz];
  int      mics2714_raw[kMqttWindowSz];
  float    bme_temp[kMqttWindowSz];
  float    bme_humidity[kMqttWindowSz];
  int32_t  bme_pressure[kMqttWindowSz];
  int      ens_aqi[kMqttWindowSz];
  int      ens_tvoc_ppb[kMqttWindowSz];
  int      ens_eco2_ppm[kMqttWindowSz];
  // Device's own verdict at the moment this window was handed off.
  uint8_t  severity;
  float    ae_mse;
  bool     verdict_ready;
};

// Small MQTT message (inference result, training log).
// Published by mqtt_task; enqueued by infer/train tasks without network I/O.
struct MqttSmallMsg {
  char topic[48];
  char payload[512];   // 512: a `latest` command reply carries all 15 channels
};

// Queue of MqttWindowData* pointers (depth 2: sensor window).
static QueueHandle_t s_mqtt_queue = nullptr;

// Queue of MqttSmallMsg values (depth 8: infer + train results, command replies).
static QueueHandle_t s_mqtt_msg_queue = nullptr;


// PSRAM-backed static pool (2 slots, ping-pong) for MqttWindowData.
// Permanently allocated once; eliminates 60-s malloc/free fragmentation.
static MqttWindowData* s_wd_pool[2] = {nullptr, nullptr};

// Mutex protecting SD card SPI access (shared between comm_task/training_task).
static SemaphoreHandle_t s_sd_mutex = nullptr;

// True only after arduino_sensors calls pipeline_notify_sd_ok().
// Guards all SD access; avoids calling SD.cardType() after a failed init.
static bool s_sd_ok = false;
static uint8_t s_sd_open_fail_streak = 0;
constexpr uint8_t kSdOpenFailMaxStreak = 5;

// ---------------------------------------------------------------------------
// Shared state between tasks.
// ---------------------------------------------------------------------------
struct Context {
  SemaphoreHandle_t weights_mutex      = nullptr;
  SemaphoreHandle_t ml_mutex           = nullptr;  // Serializes all AIfES calls (train/infer/quantize).
  float*            active_weights     = nullptr;
  float*            staging_weights    = nullptr;
  uint32_t          weight_count       = 0;
  uint8_t*          active_weights_q7  = nullptr;
  uint8_t*          staging_weights_q7 = nullptr;
  uint32_t          weight_bytes_q7    = 0;
  bool              q7_ready           = false;
  uint32_t          model_version      = 0;
  bool              weights_preloaded  = false;  // true → skip Glorot on first train
  TaskHandle_t      infer_task_handle  = nullptr;

  SemaphoreHandle_t sample_mutex = nullptr;
  float             latest_sample[kSampleDim] = {};
  float*            ring         = nullptr;  // [kRingCapacity x kSampleDim], PSRAM
  uint32_t          ring_head    = 0;
  uint32_t          ring_count   = 0;
  bool              anom_since_last_push = false;  // set by infer, cleared by comm on ring push

  // Snapshot of the newest raw sample + sample counter, for on-demand pull
  // queries (TinyAE/cmd). Written by comm_task, read by the command handler.
  tinyae::data::SensorReading latest_reading;
  bool                        latest_valid = false;
  uint32_t                    sensor_count = 0;

  // Newest on-device inference verdict. Written by inference_task; read by
  // comm_task (to stamp the sensor window) and by the command handler.
  uint8_t  latest_severity   = 0;      // AnomalySeverity value
  float    latest_mse        = 0.0f;
  bool     verdict_ready     = false;  // thresholds exist -> severity is meaningful
  uint32_t baseline_progress = 0;      // normal-MSE samples collected so far
  uint32_t baseline_target   = 0;      // samples needed before verdict_ready
};

Context g_ctx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void* malloc_psram_prefer(size_t bytes) {
  void* ptr = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ptr != nullptr) return ptr;
  return malloc(bytes);
}

// ---------------------------------------------------------------------------
// SD model persistence helpers.
// Binary format: 4-byte magic | 4-byte weight_count | weight_count × float
// ---------------------------------------------------------------------------
static constexpr uint32_t kModelFileMagic      = 0x54414503u;  // 'TAE\x03'
static constexpr uint32_t kSaveIntervalMs      = 10u * 60u * 1000u;  // >= 10 min apart
static const char* kModelFilePath              = "/models/ae_model.bin";
// 8.3-safe temp name (FatFs LFN is disabled): written first, then renamed.
static const char* kModelTmpPath               = "/models/ae_model.tmp";

// Save weights to /ae_model.bin. Call with staging_weights BEFORE the swap
// (staging = just-trained model; no mutex needed — only training_task touches it here).
static void sd_save_model(const float* weights, uint32_t weight_count, uint32_t round) {
  if (!s_sd_ok || s_sd_mutex == nullptr) return;
  if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(500)) != pdTRUE) return;

  // Write to a temp file and rename, so losing power mid-write cannot destroy
  // the previously saved model.
  File f = SD.open(kModelTmpPath, "w");  // "w" = create/truncate
  if (!f) {
    ESP_LOGW(TAG, "[SD] Cannot open %s for write", kModelTmpPath);
    xSemaphoreGive(s_sd_mutex);
    return;
  }

  const size_t nbytes = (size_t)weight_count * sizeof(float);
  bool ok = (f.write((const uint8_t*)&kModelFileMagic, 4) == 4);
  ok = ok && (f.write((const uint8_t*)&weight_count, 4) == 4);
  ok = ok && (f.write((const uint8_t*)weights, nbytes) == nbytes);
  f.close();

  if (!ok) {
    ESP_LOGW(TAG, "[SD] Model write incomplete -> previous file kept");
    SD.remove(kModelTmpPath);
    xSemaphoreGive(s_sd_mutex);
    return;
  }

  SD.remove(kModelFilePath);   // rename() will not overwrite an existing target
  if (SD.rename(kModelTmpPath, kModelFilePath)) {
    ESP_LOGI(TAG, "[SD] Model saved (round=%u, %u floats)",
             (unsigned)round, (unsigned)weight_count);
  } else {
    ESP_LOGW(TAG, "[SD] Model rename failed (temp kept at %s)", kModelTmpPath);
  }
  xSemaphoreGive(s_sd_mutex);
}

// Load /ae_model.bin into active_weights and staging_weights.
// Returns true on success.
static bool sd_load_model(float* active_w, float* staging_w, uint32_t weight_count) {
  if (!s_sd_ok || s_sd_mutex == nullptr) return false;
  if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(2000)) != pdTRUE) return false;

  bool ok = false;
  File f = SD.open(kModelFilePath, "r");
  if (!f) {
    ESP_LOGI(TAG, "[SD] No model file (%s) → fresh start", kModelFilePath);
  } else {
    uint32_t magic = 0, count = 0;
    const bool hdr_ok =
        f.read((uint8_t*)&magic, 4) == 4 && magic == kModelFileMagic &&
        f.read((uint8_t*)&count, 4) == 4 && count == weight_count;
    if (hdr_ok) {
      const size_t nbytes = weight_count * sizeof(float);
      if ((size_t)f.read((uint8_t*)active_w, nbytes) == nbytes) {
        memcpy(staging_w, active_w, nbytes);
        ESP_LOGI(TAG, "[SD] Model loaded (%u weights)", (unsigned)count);
        ok = true;
      } else {
        ESP_LOGW(TAG, "[SD] Model read incomplete → fresh start");
      }
    } else {
      ESP_LOGW(TAG, "[SD] Model invalid (magic=0x%08X count=%u) → fresh start",
               (unsigned)magic, (unsigned)count);
    }
    f.close();
  }

  xSemaphoreGive(s_sd_mutex);
  return ok;
}

// Format time_t as "YYYY-MM-DD HH:MM:SS" (KST) into buf (>=20 bytes).
// Returns buf for use in snprintf chains.
static char* fmt_ts(time_t ts, char* buf, size_t len) {
  struct tm t;
  localtime_r(&ts, &t);
  strftime(buf, len, "%Y-%m-%d %H:%M:%S", &t);
  return buf;
}

static String jnum(float v, int d = 2) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  return String(v, d);
}

static String get_mac_id() {
  uint8_t mac[6];
  WiFi.macAddress(mac);
  char buf[13];
  for (int i = 0; i < 6; i++) sprintf(&buf[i * 2], "%02X", mac[i]);
  return String(buf);
}

static String yyyymmdd_kst(time_t t) {
  struct tm kst;
  localtime_r(&t, &kst);
  char buf[9];
  strftime(buf, sizeof(buf), "%Y%m%d", &kst);
  return String(buf);
}

// Builds the JSON payload for a 60-sample MQTT window into a pre-allocated String.
// Caller must own `p` as a long-lived static to avoid repeated DRAM alloc/free.
static void build_payload(
    String& p,
    time_t start_ts, time_t end_ts, uint32_t event_seq, const String& mac_id,
    const float*   temperature_arr,
    const float*   humidity_arr,
    const int*     co2_arr,
    const int*     pm10_arr,
    const int*     pm25_arr,
    const int*     mp801_raw_arr,
    const int*     mq131_o3_ppb_arr,
    const int*     co_sen0564_raw_arr,
    const int*     mics2714_raw_arr,
    const float*   bme_temp_arr,
    const float*   bme_humidity_arr,
    const int32_t* bme_pressure_arr,
    const int*     ens_aqi_arr,
    const int*     ens_tvoc_ppb_arr,
    const int*     ens_eco2_ppm_arr,
    uint8_t        severity,
    float          ae_mse,
    bool           verdict_ready) {
  p = "";

  char seq_buf[8];
  snprintf(seq_buf, sizeof(seq_buf), "%04u", (unsigned)(event_seq % 10000));

  p += "{\"schema_version\":\"1.0\",";
  p += "\"id\":\"sen_"; p += mac_id; p += ":";
  p += String((uint32_t)start_ts); p += "-"; p += String((uint32_t)end_ts); p += "\",";
  p += "\"event_id\":\"evt_"; p += yyyymmdd_kst(start_ts); p += "_"; p += seq_buf; p += "\",";
  p += "\"modality\":\"sensor\",";
  p += "\"window\":{\"start\":"; p += String((uint32_t)start_ts);
  p += ",\"end\":"; p += String((uint32_t)end_ts);
  p += ",\"size\":"; p += String(kMqttWindowSz); p += "},";
  p += "\"metadata\":{\"feature_dim\":15,";
  p += "\"location\":{\"lat\":37.5663,\"lon\":126.9432},";
  // Device verdict (on-device TinyML). While the baseline is still building,
  // severity reports "warming" and anomaly_flag stays false.
  static const char* kSevNames[] = {"normal", "warning", "anomaly", "severe"};
  const char* sev_name = verdict_ready
                         ? kSevNames[(severity < 4) ? severity : 0]
                         : "warming";
  p += "\"anomaly_flag\":";
  p += (verdict_ready && severity >= 2) ? "true" : "false";
  p += ",\"severity\":\""; p += sev_name; p += "\",";
  p += "\"ae_mse\":"; p += String(ae_mse, 6); p += ",";
  p += "\"disaster_type\":{\"main_tag\":\"fire\",\"sub_tag\":[\"wild fire\"]},";
  p += "\"sensor\":{\"device_id\":\""; p += mac_id; p += "\",\"sampling_hz\":1,";
  p += "\"units\":{\"Temperature\":\"C\",\"Humidity\":\"%\",\"CO2\":\"ppm\",";
  p += "\"PM10\":\"ug/m3\",\"PM2.5\":\"ug/m3\",\"MP801_raw\":\"analog_val\",";
  p += "\"MQ131_O3_ppb\":\"ppb\",\"CO_SEN0564_raw\":\"analog_val\",";
  p += "\"MiCS2714_raw\":\"analog_val\",\"BME_Temp\":\"C\",";
  p += "\"BME_Humidity\":\"%\",\"BME_Pressure_Pa\":\"Pa\",";
  p += "\"ENS_AQI\":\"level_1_5\",\"ENS_TVOC_ppb\":\"ppb\",";
  p += "\"ENS_eCO2_ppm\":\"ppm\"}},";
  p += "\"updated_at\":"; p += String((uint32_t)time(nullptr)); p += "},";
  p += "\"data\":{\"blob_uri\":\"\",";

  auto arr_f = [&](const char* key, const float* arr) {
    p += "\""; p += key; p += "\":[";
    for (uint32_t i = 0; i < kMqttWindowSz; i++) {
      p += jnum(arr[i], 2);
      if (i < kMqttWindowSz - 1) p += ",";
    }
    p += "],";
  };
  auto arr_i = [&](const char* key, const int* arr) {
    p += "\""; p += key; p += "\":[";
    for (uint32_t i = 0; i < kMqttWindowSz; i++) {
      p += String(arr[i]);
      if (i < kMqttWindowSz - 1) p += ",";
    }
    p += "],";
  };

  arr_f("Temperature",    temperature_arr);
  arr_f("Humidity",       humidity_arr);
  arr_i("CO2",            co2_arr);
  arr_i("PM10",           pm10_arr);
  arr_i("PM2.5",          pm25_arr);
  arr_i("MP801_raw",      mp801_raw_arr);
  arr_i("MQ131_O3_ppb",   mq131_o3_ppb_arr);
  arr_i("CO_SEN0564_raw", co_sen0564_raw_arr);
  arr_i("MiCS2714_raw",   mics2714_raw_arr);
  arr_f("BME_Temp",       bme_temp_arr);
  arr_f("BME_Humidity",   bme_humidity_arr);

  // BME_Pressure_Pa (int32_t).
  p += "\"BME_Pressure_Pa\":[";
  for (uint32_t i = 0; i < kMqttWindowSz; i++) {
    p += String(bme_pressure_arr[i]);
    if (i < kMqttWindowSz - 1) p += ",";
  }
  p += "],";

  arr_i("ENS_AQI",      ens_aqi_arr);
  arr_i("ENS_TVOC_ppb", ens_tvoc_ppb_arr);

  // ENS_eCO2_ppm: last field, no trailing comma.
  p += "\"ENS_eCO2_ppm\":[";
  for (uint32_t i = 0; i < kMqttWindowSz; i++) {
    p += String(ens_eco2_ppm_arr[i]);
    if (i < kMqttWindowSz - 1) p += ",";
  }
  p += "]}}";
}

// ---------------------------------------------------------------------------
// SD logging helpers (mutex-protected; non-blocking on timeout).
// Files are named by date (KST) with FAT 8.3-compatible names:
// sYYMMDD.csv / tYYMMDD.csv.
// A header row is written automatically when the file is first created.
// ---------------------------------------------------------------------------
static void sd_write_sensor(const tinyae::data::SensorReading& r, time_t ts) {
  if (!s_sd_ok || s_sd_mutex == nullptr) return;
  // Use a short timeout so comm_task 1 Hz loop is not delayed.
  if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(100)) != pdTRUE) return;

  char fname[32];
  if (ts >= 1700000000) {
    struct tm t;
    localtime_r(&ts, &t);
    strftime(fname, sizeof(fname), "/datas/s%y%m%d.csv", &t);
  } else {
    strcpy(fname, "/datas/snotime.csv");
  }

  File f = SD.open(fname, FILE_APPEND);
  if (!f) {
    if (s_sd_open_fail_streak < 255) s_sd_open_fail_streak++;
    const int err = errno;
    if (s_sd_open_fail_streak >= kSdOpenFailMaxStreak) {
      ESP_LOGW(TAG, "[SD] fopen(%s) failed %u times (errno=%d: %s) -> disable SD logging",
               fname, (unsigned)s_sd_open_fail_streak, err, strerror(err));
      s_sd_ok = false;
    } else {
      ESP_LOGW(TAG, "[SD] fopen(%s) failed (%u/%u, errno=%d: %s)",
               fname, (unsigned)s_sd_open_fail_streak,
               (unsigned)kSdOpenFailMaxStreak, err, strerror(err));
    }
    xSemaphoreGive(s_sd_mutex);
    return;
  }

  if (s_sd_open_fail_streak > 0) {
    ESP_LOGI(TAG, "[SD] file open recovered");
    s_sd_open_fail_streak = 0;
  }
  if (f.size() == 0) {
    f.println("timestamp,temperature,humidity,co2,pm10,pm25,"
              "mp801_raw,mq131_o3_ppb,co_sen0564_raw,mics2714_raw,"
              "bme_temp,bme_humidity,bme_pressure,"
              "ens_aqi,ens_tvoc_ppb,ens_eco2_ppm");
  }
  char tsbuf[20];
  fmt_ts(ts, tsbuf, sizeof(tsbuf));
  char line[256];
  snprintf(line, sizeof(line),
           "%s,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%ld,%d,%d,%d",
           tsbuf,
           r.temperature, r.humidity,
           r.co2, r.pm10, r.pm25,
           r.mp801_raw, r.mq131_o3_ppb, r.co_sen0564_raw, r.mics2714_raw,
           r.bme_temp, r.bme_humidity, (long)r.bme_pressure,
           r.ens_aqi, r.ens_tvoc_ppb, r.ens_eco2_ppm);
  f.println(line);
  f.close();
  xSemaphoreGive(s_sd_mutex);
}

static void sd_write_train(uint32_t round, uint32_t version,
                           float mse, bool q7_ok, time_t ts) {
  if (!s_sd_ok || s_sd_mutex == nullptr) return;
  if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(200)) != pdTRUE) return;

  char fname[32];
  if (ts >= 1700000000) {
    struct tm t;
    localtime_r(&ts, &t);
    strftime(fname, sizeof(fname), "/datas/t%y%m%d.csv", &t);
  } else {
    strcpy(fname, "/datas/tnotime.csv");
  }

  File f = SD.open(fname, FILE_APPEND);
  if (!f) {
    if (s_sd_open_fail_streak < 255) s_sd_open_fail_streak++;
    const int err = errno;
    if (s_sd_open_fail_streak >= kSdOpenFailMaxStreak) {
      ESP_LOGW(TAG, "[SD] fopen(%s) failed %u times (errno=%d: %s) -> disable SD logging",
               fname, (unsigned)s_sd_open_fail_streak, err, strerror(err));
      s_sd_ok = false;
    } else {
      ESP_LOGW(TAG, "[SD] fopen(%s) failed (%u/%u, errno=%d: %s)",
               fname, (unsigned)s_sd_open_fail_streak,
               (unsigned)kSdOpenFailMaxStreak, err, strerror(err));
    }
    xSemaphoreGive(s_sd_mutex);
    return;
  }

  if (s_sd_open_fail_streak > 0) {
    ESP_LOGI(TAG, "[SD] file open recovered");
    s_sd_open_fail_streak = 0;
  }
  if (f.size() == 0) {
    f.println("timestamp,round,version,mse,q7_ok");
  }
  char tsbuf[20];
  fmt_ts(ts, tsbuf, sizeof(tsbuf));
  char line[100];
  snprintf(line, sizeof(line),
           "%s,%u,%u,%.8f,%d",
           tsbuf, round, version, mse, q7_ok ? 1 : 0);
  f.println(line);
  f.close();
  xSemaphoreGive(s_sd_mutex);
}

static void sd_write_infer(uint32_t version, float mse,
                           bool have_thr, float thr_warn, float thr_anom, bool anomaly,
                           time_t ts) {
  if (!s_sd_ok || s_sd_mutex == nullptr) return;
  if (xSemaphoreTake(s_sd_mutex, pdMS_TO_TICKS(200)) != pdTRUE) return;

  char fname[32];
  if (ts >= 1700000000) {
    struct tm t;
    localtime_r(&ts, &t);
    strftime(fname, sizeof(fname), "/datas/i%y%m%d.csv", &t);
  } else {
    strcpy(fname, "/datas/inotime.csv");
  }

  File f = SD.open(fname, FILE_APPEND);
  if (!f) {
    if (s_sd_open_fail_streak < 255) s_sd_open_fail_streak++;
    xSemaphoreGive(s_sd_mutex);
    return;
  }
  if (s_sd_open_fail_streak > 0) {
    s_sd_open_fail_streak = 0;
  }
  if (f.size() == 0) {
    f.println("timestamp,version,mse,thr_warn,thr_anom,anomaly");
  }
  char tsbuf[20];
  fmt_ts(ts, tsbuf, sizeof(tsbuf));
  char line[120];
  snprintf(line, sizeof(line),
           "%s,%u,%.8f,%.6f,%.6f,%d",
           tsbuf, version, mse,
           have_thr ? thr_warn : NAN, have_thr ? thr_anom : NAN,
           anomaly ? 1 : 0);
  f.println(line);
  f.close();
  xSemaphoreGive(s_sd_mutex);
}

// ---------------------------------------------------------------------------
// Anomaly severity levels based on MSE vs. IQR (Tukey fence) thresholds.
//
//  Q1 = 25th percentile, Q3 = 75th percentile of normal MSE buffer.
//  IQR = Q3 - Q1
//
//  Normal  : mse ≤ Q3 + 1.5×IQR   – within Tukey inner fence (~0.7% FP)
//  Warning : Q3+1.5×IQR < mse ≤ Q3+3.0×IQR  – mild outlier (~0.01% FP)
//  Anomaly : Q3+3.0×IQR < mse ≤ Q3+4.5×IQR  – confirmed outlier
//  Severe  : mse > Q3 + 4.5×IQR   – extreme outlier
//
// ---------------------------------------------------------------------------
enum class AnomalySeverity : uint8_t { Normal = 0, Warning = 1, Anomaly = 2, Severe = 3 };

static const char* severity_str(AnomalySeverity s) {
  switch (s) {
    case AnomalySeverity::Warning: return "warning";
    case AnomalySeverity::Anomaly: return "anomaly";
    case AnomalySeverity::Severe:  return "severe";
    default:                       return "normal";
  }
}

// ---------------------------------------------------------------------------
// Placeholder anomaly event handler.
// Called from inference_task whenever severity > Normal.
//
// TODO: extend this function to transmit the event to a remote server:
//   • HTTP POST to a REST endpoint, or
//   • publish to a dedicated MQTT topic (e.g. "TinyAE/alert").
//
// Suggested JSON payload shape:
// {
//   "type": "anomaly_event",
//   "severity": "anomaly",          // "warning" | "anomaly" | "severe"
//   "timestamp": 1772517600,        // Unix epoch (UTC)
//   "device_id": "AABBCCDDEEFF",
//   "model_version": 3,
//   "mse": 0.012345,
//   "thresholds": {
//     "warn":   0.0042,             // Q3 + 1.5*IQR
//     "anom":   0.0058,             // Q3 + 3.0*IQR
//     "severe": 0.0074              // Q3 + 4.5*IQR
//   }
// }
// ---------------------------------------------------------------------------
static void on_anomaly_event(AnomalySeverity severity,
                             uint32_t version, float mse,
                             float thr_warn, float thr_anom, float thr_severe,
                             time_t ts) {
  ESP_LOGW(TAG, "[ANOMALY_EVENT] severity=%s v=%u mse=%.6f "
                "thresholds(warn=%.6f anom=%.6f severe=%.6f) ts=%lu",
           severity_str(severity), (unsigned)version, mse,
           thr_warn, thr_anom, thr_severe, (unsigned long)ts);

  // Immediate push on a dedicated topic so the edge does not have to wait for
  // the next periodic window. Enqueued for mqtt_task — this runs on
  // inference_task and must not touch the network directly.
  if (s_mqtt_msg_queue == nullptr) return;

  MqttSmallMsg msg;
  strncpy(msg.topic, tinyae::app::kMqttTopicAlert, sizeof(msg.topic) - 1);
  msg.topic[sizeof(msg.topic) - 1] = '\0';
  snprintf(msg.payload, sizeof(msg.payload),
           "{\"type\":\"anomaly_event\",\"severity\":\"%s\",\"ts\":%lu,"
           "\"v\":%u,\"mse\":%.6f,"
           "\"thresholds\":{\"warn\":%.6f,\"anom\":%.6f,\"severe\":%.6f}}",
           severity_str(severity), (unsigned long)ts, (unsigned)version, mse,
           thr_warn, thr_anom, thr_severe);
  xQueueSend(s_mqtt_msg_queue, &msg, 0);   // drop if full; never block
}

// ---------------------------------------------------------------------------
// comm_task – 1 Hz sensor sampling, sliding-window stats AE feed, MQTT enqueue.
//
// History buffer: circular, 300 s.
// Training sample: stats over the latest 300-s window, pushed every 150 s (stride).
// Inference:       stats over available window (≥60 s), triggered every 60 s.
// No network I/O here; payload is handed off to mqtt_task via queue.
// ---------------------------------------------------------------------------
void comm_task(void* parameter) {
  auto* ctx = static_cast<Context*>(parameter);

  // Static state (BSS zero-initialized; s_event_seq starts at 1).
  // s_feat_hist: circular row-major [kStatsWindowSz][kFeatureDim].
  static float    s_feat_hist[kStatsWindowSz][kFeatureDim];
  // s_flat_hist: chronological flattening of s_feat_hist (avoids 12 KB on stack).
  static float    s_flat_hist[kStatsWindowSz * kFeatureDim];
  static uint32_t s_sensor_count;

  // MQTT window accumulation arrays (raw, unnormalized values).
  static float   s_temperature_arr[kMqttWindowSz];
  static float   s_humidity_arr[kMqttWindowSz];
  static int     s_co2_arr[kMqttWindowSz];
  static int     s_pm10_arr[kMqttWindowSz];
  static int     s_pm25_arr[kMqttWindowSz];
  static int     s_mp801_raw_arr[kMqttWindowSz];
  static int     s_mq131_o3_ppb_arr[kMqttWindowSz];
  static int     s_co_sen0564_raw_arr[kMqttWindowSz];
  static int     s_mics2714_raw_arr[kMqttWindowSz];
  static float   s_bme_temp_arr[kMqttWindowSz];
  static float   s_bme_humidity_arr[kMqttWindowSz];
  static int32_t s_bme_pressure_arr[kMqttWindowSz];
  static int     s_ens_aqi_arr[kMqttWindowSz];
  static int     s_ens_tvoc_ppb_arr[kMqttWindowSz];
  static int     s_ens_eco2_ppm_arr[kMqttWindowSz];
  static uint32_t s_mqtt_idx;
  static time_t   s_mqtt_window_start;
  static uint32_t s_event_seq = 1;

  TickType_t last_wake = xTaskGetTickCount();

  for (;;) {
    // 1. Read sensors.
    tinyae::data::SensorReading r = tinyae::data::sensors_read();
    float features[kFeatureDim] = {};
    tinyae::data::sensor_to_feature_vec(r, features);

    ESP_LOGI(TAG, "[SENSOR] T=%.1f H=%.1f CO2=%d PM10=%d PM2.5=%d O3=%dppb "
                  "MP801=%d CO=%d MiCS=%d | BME T=%.1f H=%.1f P=%ld | "
                  "ENS AQI=%d TVOC=%d eCO2=%d",
             r.temperature, r.humidity, r.co2, r.pm10, r.pm25,
             r.mq131_o3_ppb, r.mp801_raw, r.co_sen0564_raw, r.mics2714_raw,
             r.bme_temp, r.bme_humidity, (long)r.bme_pressure,
             r.ens_aqi, r.ens_tvoc_ppb, r.ens_eco2_ppm);

    // 2. Write normalized features into circular history buffer.
    const uint32_t h_pos = s_sensor_count % kStatsWindowSz;
    memcpy(s_feat_hist[h_pos], features, sizeof(float) * kFeatureDim);
    s_sensor_count++;

    // Snapshot newest raw sample for on-demand pull queries (TinyAE/cmd).
    xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
    ctx->latest_reading = r;
    ctx->latest_valid   = true;
    ctx->sensor_count   = s_sensor_count;
    xSemaphoreGive(ctx->sample_mutex);

    // 3. Check whether inference or training should fire this tick.
    //    avail = samples available (grows until kStatsWindowSz, then stays there).
    //    oldest = circular buffer start index for chronological flattening.
    const uint32_t avail  = (s_sensor_count < kStatsWindowSz) ? s_sensor_count : kStatsWindowSz;
    const uint32_t oldest = (s_sensor_count >= kStatsWindowSz)
                            ? (s_sensor_count % kStatsWindowSz) : 0;

    const bool do_infer = (avail >= kInferInterval) &&
                          (s_sensor_count % kInferInterval == 0);
    const bool do_train = (avail == kStatsWindowSz) &&
                          (s_sensor_count % kTrainStride == 0);

    if (do_infer || do_train) {
      // Flatten circular buffer into chronological order once.
      for (uint32_t i = 0; i < avail; i++) {
        const uint32_t src = (oldest + i) % kStatsWindowSz;
        memcpy(&s_flat_hist[i * kFeatureDim], s_feat_hist[src], sizeof(float) * kFeatureDim);
      }

      float stats[kSampleDim];
      tinyae::data::compute_stats_vec(s_flat_hist, avail, stats);

      xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);

      if (do_train) {
        const bool skip_anom = ctx->anom_since_last_push;
        ctx->anom_since_last_push = false;  // reset for next window regardless
        if (skip_anom) {
          ESP_LOGI(TAG, "[RING] skip: anomaly detected in previous window");
        } else {
          memcpy(ctx->ring + ctx->ring_head * kSampleDim, stats, sizeof(float) * kSampleDim);
          ctx->ring_head = (ctx->ring_head + 1) % kRingCapacity;
          if (ctx->ring_count < kRingCapacity) ctx->ring_count++;
        }
      }

      if (do_infer) {
        memcpy(ctx->latest_sample, stats, sizeof(float) * kSampleDim);
      }

      xSemaphoreGive(ctx->sample_mutex);

      ESP_LOGI(TAG, "[STATS] t=%u avail=%u infer=%d train=%d | stats[0..5]=%.3f %.3f %.3f %.3f %.3f %.3f",
               s_sensor_count, avail, do_infer, do_train,
               stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);

      if (do_infer && ctx->infer_task_handle != nullptr) {
        xTaskNotifyGive(ctx->infer_task_handle);
      }
    }

    // 4. Accumulate raw values for MQTT window + write to SD card.
    time_t now = time(nullptr);
    if (now < 1700000000) now = 0;
    if (s_mqtt_idx == 0) s_mqtt_window_start = now;

    sd_write_sensor(r, now);

    // Log SD capacity every 30 min (1800 samples at 1 Hz).
    static uint32_t s_sd_cap_tick = 0;
    if (++s_sd_cap_tick >= 1800) {
      s_sd_cap_tick = 0;
      if (s_sd_ok) {
        const uint64_t total_mb = SD.totalBytes() / (1024ULL * 1024ULL);
        const uint64_t used_mb  = SD.usedBytes()  / (1024ULL * 1024ULL);
        ESP_LOGI(TAG, "[SD] %llu/%llu MB used (%.1f%%)",
                 used_mb, total_mb,
                 (total_mb > 0) ? (100.0f * (float)used_mb / (float)total_mb) : 0.0f);
      }
    }

    s_temperature_arr[s_mqtt_idx]    = r.temperature;
    s_humidity_arr[s_mqtt_idx]       = r.humidity;
    s_co2_arr[s_mqtt_idx]            = r.co2;
    s_pm10_arr[s_mqtt_idx]           = r.pm10;
    s_pm25_arr[s_mqtt_idx]           = r.pm25;
    s_mp801_raw_arr[s_mqtt_idx]      = r.mp801_raw;
    s_mq131_o3_ppb_arr[s_mqtt_idx]   = r.mq131_o3_ppb;
    s_co_sen0564_raw_arr[s_mqtt_idx] = r.co_sen0564_raw;
    s_mics2714_raw_arr[s_mqtt_idx]   = r.mics2714_raw;
    s_bme_temp_arr[s_mqtt_idx]       = r.bme_temp;
    s_bme_humidity_arr[s_mqtt_idx]   = r.bme_humidity;
    s_bme_pressure_arr[s_mqtt_idx]   = r.bme_pressure;
    s_ens_aqi_arr[s_mqtt_idx]        = r.ens_aqi;
    s_ens_tvoc_ppb_arr[s_mqtt_idx]   = r.ens_tvoc_ppb;
    s_ens_eco2_ppm_arr[s_mqtt_idx]   = r.ens_eco2_ppm;
    s_mqtt_idx++;

    // 5. When window full, hand off to mqtt_task (non-blocking, no network I/O here).
    if (s_mqtt_idx >= kMqttWindowSz) {
      // Use pre-allocated PSRAM pool slot (ping-pong); no heap churn.
      static uint8_t s_wd_idx = 0;
      MqttWindowData* wd = s_wd_pool[s_wd_idx];
      s_wd_idx ^= 1;

      if (wd != nullptr) {
        wd->window_start = s_mqtt_window_start;
        wd->event_seq    = s_event_seq;
        memcpy(wd->temperature,    s_temperature_arr,    sizeof(float)   * kMqttWindowSz);
        memcpy(wd->humidity,       s_humidity_arr,       sizeof(float)   * kMqttWindowSz);
        memcpy(wd->co2,            s_co2_arr,            sizeof(int)     * kMqttWindowSz);
        memcpy(wd->pm10,           s_pm10_arr,           sizeof(int)     * kMqttWindowSz);
        memcpy(wd->pm25,           s_pm25_arr,           sizeof(int)     * kMqttWindowSz);
        memcpy(wd->mp801_raw,      s_mp801_raw_arr,      sizeof(int)     * kMqttWindowSz);
        memcpy(wd->mq131_o3_ppb,   s_mq131_o3_ppb_arr,   sizeof(int)     * kMqttWindowSz);
        memcpy(wd->co_sen0564_raw, s_co_sen0564_raw_arr, sizeof(int)     * kMqttWindowSz);
        memcpy(wd->mics2714_raw,   s_mics2714_raw_arr,   sizeof(int)     * kMqttWindowSz);
        memcpy(wd->bme_temp,       s_bme_temp_arr,       sizeof(float)   * kMqttWindowSz);
        memcpy(wd->bme_humidity,   s_bme_humidity_arr,   sizeof(float)   * kMqttWindowSz);
        memcpy(wd->bme_pressure,   s_bme_pressure_arr,   sizeof(int32_t) * kMqttWindowSz);
        memcpy(wd->ens_aqi,        s_ens_aqi_arr,        sizeof(int)     * kMqttWindowSz);
        memcpy(wd->ens_tvoc_ppb,   s_ens_tvoc_ppb_arr,   sizeof(int)     * kMqttWindowSz);
        memcpy(wd->ens_eco2_ppm,   s_ens_eco2_ppm_arr,   sizeof(int)     * kMqttWindowSz);

        // Stamp the window with the device's own latest verdict.
        xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
        wd->severity      = ctx->latest_severity;
        wd->ae_mse        = ctx->latest_mse;
        wd->verdict_ready = ctx->verdict_ready;
        xSemaphoreGive(ctx->sample_mutex);

        if (xQueueSend(s_mqtt_queue, &wd, 0) != pdTRUE) {
          ESP_LOGW(TAG, "[MQTT] queue full, dropping window");
        }
      }
      s_mqtt_idx = 0;
      s_event_seq++;
    }

    // 6. Maintain strict 1 Hz timing (network ops no longer block this path).
    vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(1000));
  }
}

// ---------------------------------------------------------------------------
// On-demand pull: commands arrive on TinyAE/cmd/<MAC> (or /all); the reply is
// published to TinyAE/resp/<MAC>. This runs inside client.loop() on mqtt_task,
// so it only reads shared snapshots and enqueues — it must never block.
// Accepts a bare word ("status") or {"cmd":"status"}.
// ---------------------------------------------------------------------------
static String s_mac_id;
static char   s_resp_topic[64] = {0};

static void mqtt_command_cb(char* /*topic*/, uint8_t* payload, unsigned int length) {
  char buf[96];
  const unsigned int n = (length < sizeof(buf) - 1) ? length : sizeof(buf) - 1;
  memcpy(buf, payload, n);
  buf[n] = '\0';

  const char* cmd = buf;
  const char* q   = strstr(buf, "\"cmd\"");
  if (q != nullptr) {
    q = strchr(q + 5, '"');
    if (q != nullptr) cmd = q + 1;
  }

  MqttSmallMsg msg;
  strncpy(msg.topic, s_resp_topic, sizeof(msg.topic) - 1);
  msg.topic[sizeof(msg.topic) - 1] = '\0';

  if (strncmp(cmd, "ping", 4) == 0) {
    snprintf(msg.payload, sizeof(msg.payload),
             "{\"dev\":\"%s\",\"cmd\":\"ping\",\"up\":%lu}",
             s_mac_id.c_str(), (unsigned long)(millis() / 1000));

  } else if (strncmp(cmd, "status", 6) == 0) {
    uint32_t version = 0, n_samples = 0;
    bool     preloaded = false;
    if (xSemaphoreTake(g_ctx.weights_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
      version   = g_ctx.model_version;
      preloaded = g_ctx.weights_preloaded;
      xSemaphoreGive(g_ctx.weights_mutex);
    }
    if (xSemaphoreTake(g_ctx.sample_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
      n_samples = g_ctx.sensor_count;
      xSemaphoreGive(g_ctx.sample_mutex);
    }
    snprintf(msg.payload, sizeof(msg.payload),
             "{\"dev\":\"%s\",\"cmd\":\"status\",\"up\":%lu,\"v\":%u,\"pre\":%d,"
             "\"n\":%u,\"sd\":%d,\"heap\":%u,\"psram\":%u,\"rssi\":%d,\"r0\":%.2f}",
             s_mac_id.c_str(), (unsigned long)(millis() / 1000), (unsigned)version,
             preloaded ? 1 : 0, (unsigned)n_samples, s_sd_ok ? 1 : 0,
             (unsigned)ESP.getFreeHeap(),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (int)WiFi.RSSI(), tinyae::data::sensors_mq131_r0());

  } else if (strncmp(cmd, "latest", 6) == 0) {
    tinyae::data::SensorReading r;
    bool valid = false;
    if (xSemaphoreTake(g_ctx.sample_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
      r     = g_ctx.latest_reading;
      valid = g_ctx.latest_valid;
      xSemaphoreGive(g_ctx.sample_mutex);
    }
    if (!valid) {
      snprintf(msg.payload, sizeof(msg.payload),
               "{\"dev\":\"%s\",\"cmd\":\"latest\",\"err\":\"no sample yet\"}",
               s_mac_id.c_str());
    } else {
      snprintf(msg.payload, sizeof(msg.payload),
               "{\"dev\":\"%s\",\"cmd\":\"latest\",\"t\":%.2f,\"h\":%.2f,\"co2\":%d,"
               "\"pm10\":%d,\"pm25\":%d,\"mp801\":%d,\"o3\":%d,\"co\":%d,\"mics\":%d,"
               "\"bt\":%.2f,\"bh\":%.2f,\"bp\":%ld,\"aqi\":%d,\"tvoc\":%d,\"eco2\":%d}",
               s_mac_id.c_str(), r.temperature, r.humidity, r.co2, r.pm10, r.pm25,
               r.mp801_raw, r.mq131_o3_ppb, r.co_sen0564_raw, r.mics2714_raw,
               r.bme_temp, r.bme_humidity, (long)r.bme_pressure,
               r.ens_aqi, r.ens_tvoc_ppb, r.ens_eco2_ppm);
    }

  } else {
    snprintf(msg.payload, sizeof(msg.payload),
             "{\"dev\":\"%s\",\"err\":\"unknown cmd\","
             "\"accepts\":[\"ping\",\"status\",\"latest\"]}", s_mac_id.c_str());
  }

  if (s_mqtt_msg_queue != nullptr) xQueueSend(s_mqtt_msg_queue, &msg, 0);
}

// ---------------------------------------------------------------------------
// mqtt_task – handles all network I/O: WiFi/MQTT reconnect, keepalive, publish.
// Runs independently of the 1 Hz sensor loop; never affects sensor timing.
// ---------------------------------------------------------------------------
void mqtt_task(void* /*parameter*/) {
  WiFiClient   net;
  PubSubClient client(net);
  client.setServer(tinyae::app::kMqttHost, tinyae::app::kMqttPort);
  client.setBufferSize(10000);

  const String mac_id = get_mac_id();
  char client_id[32];
  snprintf(client_id, sizeof(client_id), "TinyAE_%s", mac_id.c_str());
  ESP_LOGI(TAG, "mqtt_task start: clientId=%s", client_id);

  // On-demand pull wiring: listen on <cmd>/<MAC> and <cmd>/all, reply on <resp>/<MAC>.
  char cmd_topic[64], cmd_all[64];
  snprintf(cmd_topic, sizeof(cmd_topic), "%s/%s", tinyae::app::kMqttTopicCmd, mac_id.c_str());
  snprintf(cmd_all,   sizeof(cmd_all),   "%s/all", tinyae::app::kMqttTopicCmd);
  snprintf(s_resp_topic, sizeof(s_resp_topic), "%s/%s",
           tinyae::app::kMqttTopicResp, mac_id.c_str());
  s_mac_id = mac_id;
  client.setCallback(mqtt_command_cb);

  // Pre-allocate payload String once; reused every 60 s (no DRAM heap churn).
  static String s_payload;
  s_payload.reserve(9000);

  for (;;) {
    // Block up to 1 s for new data; the idle time drives keepalive and, just as
    // importantly, client.loop() — which is where inbound commands are dispatched.
    MqttWindowData* wd = nullptr;
    const BaseType_t got = xQueueReceive(s_mqtt_queue, &wd, pdMS_TO_TICKS(1000));

    // Reconnect WiFi only when in a terminal failure state (not while connecting).
    // Rate-limited to one attempt per 15 s to avoid rapid disconnect/begin churn.
    {
      static TickType_t s_last_wifi_begin = 0;
      const uint8_t wifi_st = WiFi.status();
      if (wifi_st != WL_CONNECTED) {
        const TickType_t now_ticks = xTaskGetTickCount();
        if ((now_ticks - s_last_wifi_begin) >= pdMS_TO_TICKS(15000)) {
          ESP_LOGW(TAG, "[MQTT] WiFi lost (st=%u), reconnecting...", (unsigned)wifi_st);
          WiFi.begin(tinyae::app::kWifiSsid, tinyae::app::kWifiPass);
          s_last_wifi_begin = now_ticks;
        }
      }
    }

    // Reconnect MQTT if WiFi is up but broker connection dropped.
    if (WiFi.status() == WL_CONNECTED && !client.connected()) {
      if (client.connect(client_id)) {
        ESP_LOGI(TAG, "[MQTT] connected to %s:%u", tinyae::app::kMqttHost, tinyae::app::kMqttPort);
        client.subscribe(cmd_topic);
        client.subscribe(cmd_all);
        ESP_LOGI(TAG, "[MQTT] subscribed: %s | %s -> reply %s",
                 cmd_topic, cmd_all, s_resp_topic);
      } else {
        ESP_LOGW(TAG, "[MQTT] connect failed (state=%d), retry in 5 s", client.state());
      }
    }

    // Keepalive ping to broker (must be called regularly).
    if (client.connected()) {
      client.loop();
    }

    // Publish if we received a window.
    if (got == pdTRUE && wd != nullptr) {
      if (client.connected()) {
        const time_t window_end = wd->window_start + (time_t)kMqttWindowSz;
        build_payload(
            s_payload,
            wd->window_start, window_end, wd->event_seq, mac_id,
            wd->temperature, wd->humidity, wd->co2, wd->pm10, wd->pm25,
            wd->mp801_raw, wd->mq131_o3_ppb, wd->co_sen0564_raw, wd->mics2714_raw,
            wd->bme_temp, wd->bme_humidity, wd->bme_pressure,
            wd->ens_aqi, wd->ens_tvoc_ppb, wd->ens_eco2_ppm,
            wd->severity, wd->ae_mse, wd->verdict_ready);
        const bool ok = client.publish(tinyae::app::kMqttTopicSensor, s_payload.c_str());
        ESP_LOGI(TAG, "[MQTT] publish %s len=%u win=%u..%u",
                 ok ? "OK" : "FAIL", (unsigned)s_payload.length(),
                 (unsigned)wd->window_start, (unsigned)window_end);
      } else {
        ESP_LOGW(TAG, "[MQTT] not connected, dropping window");
      }
      // No free(wd): wd points into s_wd_pool[] which is permanently allocated.
    }

    // Drain small messages (infer / train results) — non-blocking.
    if (client.connected() && s_mqtt_msg_queue != nullptr) {
      MqttSmallMsg smsg;
      while (xQueueReceive(s_mqtt_msg_queue, &smsg, 0) == pdTRUE) {
        client.publish(smsg.topic, smsg.payload);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// inference_task – 1 Hz AE reconstruction, MSE-based anomaly detection.
// Improvements:
//   - Thresholds cached; sorted only when new MSE is pushed.
//   - Baseline protected: MSE > p95 not added to normal_mse_buf.
// ---------------------------------------------------------------------------
void inference_task(void* parameter) {
  auto* ctx = static_cast<Context*>(parameter);

  constexpr uint32_t kNormalMseCapacity        = 256;
  // 20 inference cycles (~20 min at kInferInterval=60 s) before thresholds are
  // trusted. Lower than 64 so a rebooted device can judge sooner; still enough
  // samples for a usable IQR.
  constexpr uint32_t kNormalMseMinForThreshold = 20;
  constexpr uint32_t kThrCount   = 3;
  constexpr uint32_t kThrWarn    = 0;  // Q3 + 1.5*IQR  (~0.7% FP)
  constexpr uint32_t kThrAnom    = 1;  // Q3 + 3.0*IQR  (~0.01% FP)
  constexpr uint32_t kThrSevere  = 2;  // Q3 + 4.5*IQR

  float    normal_mse_buf[kNormalMseCapacity] = {};
  uint32_t normal_mse_head  = 0;
  uint32_t normal_mse_count = 0;
  float    thr[kThrCount]   = {};
  bool     have_thresholds  = false;
  bool     thr_dirty        = false;  // true when buf updated, recompute needed
  uint32_t consec_anomalies = 0;      // death-spiral counter
  uint32_t prev_version     = 0;      // detect model version change
  bool     prev_used_q7     = false;  // detect F32↔Q7 inference mode change

  auto push_normal_mse = [&](float mse) {
    normal_mse_buf[normal_mse_head] = mse;
    normal_mse_head = (normal_mse_head + 1) % kNormalMseCapacity;
    if (normal_mse_count < kNormalMseCapacity) normal_mse_count++;
    thr_dirty = true;
  };

  // Returns true and fills thr[] only when there is enough data.
  // Uses cached result when thr_dirty == false.
  auto compute_thresholds = [&]() -> bool {
    if (!thr_dirty && have_thresholds) return true;
    if (normal_mse_count < kNormalMseMinForThreshold) return false;

    float sorted[kNormalMseCapacity];
    const uint32_t n = normal_mse_count;
    for (uint32_t i = 0; i < n; i++) {
      const uint32_t idx =
          (normal_mse_head + kNormalMseCapacity - n + i) % kNormalMseCapacity;
      sorted[i] = normal_mse_buf[idx];
    }
    std::sort(sorted, sorted + n);

    // IQR (Tukey fence) thresholds – robust to non-Gaussian MSE distributions.
    const float q1  = sorted[(uint32_t)((float)(n - 1) * 0.25f)];
    const float q3  = sorted[(uint32_t)((float)(n - 1) * 0.75f)];
    const float iqr = q3 - q1;
    thr[kThrWarn]   = q3 + 1.5f * iqr;   // Tukey inner fence  (~0.7% FP)
    thr[kThrAnom]   = q3 + 3.0f * iqr;   // Tukey outer fence  (~0.01% FP)
    thr[kThrSevere] = q3 + 4.5f * iqr;   // Extended outer fence
    thr_dirty       = false;
    return true;
  };

  for (;;) {
    // Wait for comm_task to push a new stats sample (kInferInterval seconds).
    // Timeout is 1.5× kInferInterval so we don't busy-loop but also recover
    // if a notification is ever missed.
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(kInferInterval * 1500));

    float input[kSampleDim]  = {};
    float output[kSampleDim] = {};

    xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
    memcpy(input, ctx->latest_sample, sizeof(float) * kSampleDim);
    xSemaphoreGive(ctx->sample_mutex);

    uint32_t version = 0;
    bool     used_q7 = false;
    int8_t   err     = 0;
    float*   weights_f32 = nullptr;
    uint8_t* weights_q7  = nullptr;

    xSemaphoreTake(ctx->weights_mutex, portMAX_DELAY);
    version     = ctx->model_version;
    used_q7     = (ctx->q7_ready && ctx->active_weights_q7 != nullptr);
    weights_f32 = ctx->active_weights;
    weights_q7  = ctx->active_weights_q7;
    xSemaphoreGive(ctx->weights_mutex);

    xSemaphoreTake(ctx->ml_mutex, portMAX_DELAY);
    if (used_q7 && weights_q7 != nullptr) {
      err = tinyae::ml::autoencoder_infer_q7(weights_q7, input, 1, output);
    } else {
      used_q7 = false;
      err = tinyae::ml::autoencoder_infer(weights_f32, input, 1, output);
    }
    xSemaphoreGive(ctx->ml_mutex);

    if (err != 0) {
      ESP_LOGE(TAG, "Inference failed (err=%d)", (int)err);
      continue;
    }

    const float mse = tinyae::ml::autoencoder_mean_reconstruction_mse(input, output, 1);

    // Reset baseline on first training start (v=0 → v>0) OR when inference mode
    // switches between F32 and Q7.  Q7 quantisation introduces a systematic MSE
    // offset (~0.088 vs ~0.0005 for F32) that would produce a bimodal baseline and
    // impossibly wide IQR thresholds if both modes share the same buffer.
    // NOTE: training_task increments model_version on every round (~10 ms), so we
    // must NOT reset on every version change — only on these two distinct transitions.
    {
      const bool mode_changed = (used_q7 != prev_used_q7);
      const bool first_train  = (prev_version == 0 && version > 0);
      if (first_train || mode_changed) {
        normal_mse_count = 0;
        normal_mse_head  = 0;
        have_thresholds  = false;
        thr_dirty        = false;
        consec_anomalies = 0;
        if (first_train)
          ESP_LOGI(TAG, "Baseline init: first training (v%u)", (unsigned)version);
        else
          ESP_LOGI(TAG, "Baseline reset: infer mode %s→%s",
                   prev_used_q7 ? "Q7" : "F32", used_q7 ? "Q7" : "F32");
      }
    }
    prev_version  = version;
    prev_used_q7  = used_q7;

    // Update baseline: skip if MSE already exceeds warn threshold (baseline protection).
    // Death-spiral guard: if kNormalMseCapacity consecutive samples were all anomalous,
    // force-push the current MSE so the baseline can drift toward the new distribution.
    if (version > 0) {
      const bool above_warn = have_thresholds && (mse > thr[kThrWarn]);
      if (!above_warn) {
        push_normal_mse(mse);
        consec_anomalies = 0;
      } else {
        consec_anomalies++;
        if (consec_anomalies >= kNormalMseCapacity) {
          push_normal_mse(mse);
          consec_anomalies = 0;
          ESP_LOGW(TAG, "Baseline drift: forced push mse=%.6f", mse);
        }
        // Signal comm_task to skip next ring push (training anomaly exclusion).
        xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
        ctx->anom_since_last_push = true;
        xSemaphoreGive(ctx->sample_mutex);
      }
    }

    // Recompute thresholds only when new data was pushed (cached otherwise).
    have_thresholds = compute_thresholds();

    const time_t now_ts = time(nullptr);

    // Classify only once thresholds are trusted; before that the device is
    // still building its normal-MSE baseline and cannot judge.
    AnomalySeverity sev     = AnomalySeverity::Normal;
    bool            anomaly = false;
    if (have_thresholds) {
      if      (mse > thr[kThrSevere]) sev = AnomalySeverity::Severe;
      else if (mse > thr[kThrAnom])   sev = AnomalySeverity::Anomaly;
      else if (mse > thr[kThrWarn])   sev = AnomalySeverity::Warning;
      else                             sev = AnomalySeverity::Normal;
      anomaly = (sev >= AnomalySeverity::Anomaly);
    }

    // Share the verdict so the 60 s sensor window and the on-demand command
    // handler both report what the device itself decided.
    xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
    ctx->latest_severity   = (uint8_t)sev;
    ctx->latest_mse        = mse;
    ctx->verdict_ready     = have_thresholds;
    ctx->baseline_progress = normal_mse_count;
    ctx->baseline_target   = kNormalMseMinForThreshold;
    xSemaphoreGive(ctx->sample_mutex);

    if (have_thresholds) {
      ESP_LOGI(TAG, "Infer(%s) v%u mse=%.6f warn=%.4f anom=%.4f [%s]",
               used_q7 ? "Q7" : "F32", (unsigned)version, mse,
               thr[kThrWarn], thr[kThrAnom], severity_str(sev));

      // Notify on any non-normal event (warning, anomaly, severe).
      if (sev > AnomalySeverity::Normal) {
        on_anomaly_event(sev, version, mse,
                         thr[kThrWarn], thr[kThrAnom], thr[kThrSevere], now_ts);
      }
    } else {
      ESP_LOGI(TAG, "Infer(%s) v%u mse=%.6f (baseline %u/%u)",
               used_q7 ? "Q7" : "F32", (unsigned)version, mse,
               (unsigned)normal_mse_count, (unsigned)kNormalMseMinForThreshold);
    }

    sd_write_infer(version, mse, have_thresholds,
                   have_thresholds ? thr[kThrWarn] : 0.0f,
                   have_thresholds ? thr[kThrAnom] : 0.0f,
                   anomaly, now_ts);

    // Publish every cycle — including during warm-up — so the edge can tell
    // "still building a baseline" apart from "device went silent".
    if (s_mqtt_msg_queue != nullptr) {
      MqttSmallMsg msg;
      strncpy(msg.topic, tinyae::app::kMqttTopicInfer, sizeof(msg.topic) - 1);
      msg.topic[sizeof(msg.topic) - 1] = '\0';
      if (have_thresholds) {
        snprintf(msg.payload, sizeof(msg.payload),
                 "{\"state\":\"ready\",\"v\":%u,\"mse\":%.6f,\"severity\":\"%s\","
                 "\"anomaly\":%s,\"warn\":%.6f,\"anom\":%.6f,\"severe\":%.6f}",
                 (unsigned)version, mse, severity_str(sev),
                 anomaly ? "true" : "false",
                 thr[kThrWarn], thr[kThrAnom], thr[kThrSevere]);
      } else {
        snprintf(msg.payload, sizeof(msg.payload),
                 "{\"state\":\"warming\",\"v\":%u,\"mse\":%.6f,"
                 "\"severity\":\"warming\",\"anomaly\":false,"
                 "\"baseline\":%u,\"baseline_target\":%u}",
                 (unsigned)version, mse,
                 (unsigned)normal_mse_count,
                 (unsigned)kNormalMseMinForThreshold);
      }
      xQueueSend(s_mqtt_msg_queue, &msg, 0);
    }
  }
}

// ---------------------------------------------------------------------------
// training_task – continuous online learning on normal ring buffer samples.
// ---------------------------------------------------------------------------
void training_task(void* parameter) {
  auto* ctx = static_cast<Context*>(parameter);

  // Allocate batch and reconstruction buffers from PSRAM
  // (each kTrainBatch x kSampleDim x 4 bytes = 16 x 300 x 4 = 19.2 KB).
  float* batch = static_cast<float*>(
      malloc_psram_prefer(kTrainBatch * kSampleDim * sizeof(float)));
  float* recon  = static_cast<float*>(
      malloc_psram_prefer(kTrainBatch * kSampleDim * sizeof(float)));

  if (batch == nullptr || recon == nullptr) {
    ESP_LOGE(TAG, "training_task: OOM allocating batch/recon");
    free(batch);
    free(recon);
    vTaskDelete(nullptr);
    return;
  }

  srand((unsigned)esp_random());

  constexpr TickType_t kTrainLogInterval = pdMS_TO_TICKS(5000);
  TickType_t last_train_log = 0;
  uint32_t   train_round    = 0;
  bool       first_run      = !ctx->weights_preloaded;  // Glorot unless loaded from SD

  tinyae::ml::AutoencoderTrainConfig cfg = {};
  cfg.epochs                     = 200;
  cfg.learn_rate                 = 0.01f;
  cfg.early_stopping_target_loss = 0.0005f;
  cfg.optimizer                  = tinyae::ml::AutoencoderOptimizer::Adam;
  cfg.sgd_momentum               = 0.0f;
  cfg.log_every_epochs           = 50;

  for (;;) {
    // Sample kTrainBatch windows uniformly at random from the ring buffer.
    // Random sampling (vs. always taking the most-recent N) gives the model
    // exposure to the full hour of history and improves generalisation.
    bool have_batch = false;
    xSemaphoreTake(ctx->sample_mutex, portMAX_DELAY);
    if (ctx->ring_count >= kTrainBatch) {
      const uint32_t oldest =
          (ctx->ring_head + kRingCapacity - ctx->ring_count) % kRingCapacity;
      for (uint32_t i = 0; i < kTrainBatch; i++) {
        const uint32_t rand_idx = (uint32_t)rand() % ctx->ring_count;
        const uint32_t ring_idx = (oldest + rand_idx) % kRingCapacity;
        memcpy(&batch[i * kSampleDim],
               ctx->ring + ring_idx * kSampleDim,
               sizeof(float) * kSampleDim);
      }
      have_batch = true;
    }
    xSemaphoreGive(ctx->sample_mutex);

    if (!have_batch) {
      vTaskDelay(pdMS_TO_TICKS(250));
      continue;
    }

    if (first_run) ESP_LOGI(TAG, "Train start (init=1)");

    // AIfES is not re-entrant; serialize train/infer/quantize calls.
    xSemaphoreTake(ctx->ml_mutex, portMAX_DELAY);

    const int8_t err = tinyae::ml::autoencoder_train(
        ctx->staging_weights, batch, kTrainBatch, cfg, first_run, recon);
    if (err != 0) {
      xSemaphoreGive(ctx->ml_mutex);
      ESP_LOGE(TAG, "Training failed (err=%d)", (int)err);
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    first_run = false;
    train_round++;

    // Evaluate on the same batch (staging weights).
    float mean_mse = 0.0f;
    bool  have_mse = false;
    if (tinyae::ml::autoencoder_infer(ctx->staging_weights, batch, kTrainBatch, recon) == 0) {
      mean_mse = tinyae::ml::autoencoder_mean_reconstruction_mse(batch, recon, kTrainBatch);
      have_mse = true;
    }

    // Optionally quantize to Q7 for lower-latency inference.
    bool q7_ok = false;
    if (ctx->staging_weights_q7 != nullptr && ctx->active_weights_q7 != nullptr) {
      const int8_t qerr = tinyae::ml::autoencoder_quantize_f32_to_q7(
          ctx->staging_weights, batch, kTrainBatch, ctx->staging_weights_q7);
      q7_ok = (qerr == 0);
      if (!q7_ok) ESP_LOGW(TAG, "Q7 quantize failed (err=%d) -> F32 fallback", (int)qerr);
    }

    xSemaphoreGive(ctx->ml_mutex);

    // Save staging_weights (just-trained model) to SD before the pointer swap.
    // staging_weights is only accessed by training_task at this point → no mutex needed.
    // Round-based saving fires every few seconds once training is hot, which
    // rewrites ~25 KB to the card far more often than useful. Save the first
    // round, then at most once per kSaveIntervalMs.
    {
      static TickType_t s_last_model_save = 0;
      const TickType_t now_ticks = xTaskGetTickCount();
      if (train_round == 1 ||
          (now_ticks - s_last_model_save) >= pdMS_TO_TICKS(kSaveIntervalMs)) {
        sd_save_model(ctx->staging_weights, ctx->weight_count, train_round);
        s_last_model_save = now_ticks;
      }
    }

    // Commit staging -> active: pointer swap keeps mutex critical section to µs.
    // After swap, staging points to the old active buffer; next training round
    // continues from those weights (warm-start).
    xSemaphoreTake(ctx->weights_mutex, portMAX_DELAY);
    std::swap(ctx->active_weights, ctx->staging_weights);
    if (q7_ok) {
      std::swap(ctx->active_weights_q7, ctx->staging_weights_q7);
      ctx->q7_ready = true;
    } else {
      ctx->q7_ready = false;
    }
    ctx->model_version++;
    xSemaphoreGive(ctx->weights_mutex);

    const TickType_t now = xTaskGetTickCount();
    if (train_round == 1 || (now - last_train_log) >= kTrainLogInterval) {
      if (have_mse) {
        ESP_LOGI(TAG, "Train round=%u -> v%u mse=%.6f q7=%d",
                 (unsigned)train_round, (unsigned)ctx->model_version, mean_mse, q7_ok ? 1 : 0);
        sd_write_train(train_round, ctx->model_version, mean_mse, q7_ok, time(nullptr));

        // Publish to TinyAE/train (non-blocking; drop if queue full).
        if (s_mqtt_msg_queue != nullptr) {
          MqttSmallMsg msg;
          strncpy(msg.topic, tinyae::app::kMqttTopicTrain, sizeof(msg.topic) - 1);
          msg.topic[sizeof(msg.topic) - 1] = '\0';
          snprintf(msg.payload, sizeof(msg.payload),
                   "{\"round\":%u,\"v\":%u,\"mse\":%.6f,\"q7\":%d}",
                   (unsigned)train_round, (unsigned)ctx->model_version,
                   mean_mse, q7_ok ? 1 : 0);
          xQueueSend(s_mqtt_msg_queue, &msg, 0);
        }
      } else {
        ESP_LOGI(TAG, "Train round=%u -> v%u mse=(n/a) q7=%d",
                 (unsigned)train_round, (unsigned)ctx->model_version, q7_ok ? 1 : 0);
      }
      last_train_log = now;
    }

    // Yield to other tasks.
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
namespace tinyae {
namespace app {

void start_pipeline(const PipelineConfig& cfg) {
  static bool started = false;
  if (started) return;
  started = true;

  ESP_LOGI(TAG, "Pipeline mode: %s",
           cfg.mode == PipelineMode::DataCollection ? "DataCollection" : "Training");

  g_ctx.weights_mutex = xSemaphoreCreateMutex();
  g_ctx.ml_mutex      = xSemaphoreCreateMutex();
  g_ctx.sample_mutex  = xSemaphoreCreateMutex();
  s_sd_mutex          = xSemaphoreCreateMutex();

  g_ctx.weight_count = tinyae::ml::autoencoder_weight_count();
  const size_t weight_bytes = sizeof(float) * g_ctx.weight_count;

  g_ctx.active_weights     = static_cast<float*>(malloc_psram_prefer(weight_bytes));
  g_ctx.staging_weights    = static_cast<float*>(malloc_psram_prefer(weight_bytes));
  g_ctx.weight_bytes_q7    = tinyae::ml::autoencoder_weight_bytes_q7();
  g_ctx.active_weights_q7  = static_cast<uint8_t*>(malloc_psram_prefer(g_ctx.weight_bytes_q7));
  g_ctx.staging_weights_q7 = static_cast<uint8_t*>(malloc_psram_prefer(g_ctx.weight_bytes_q7));

  // Ring buffer: kRingCapacity x kSampleDim floats (3600 x 300 x 4 = 4.1 MB), on PSRAM.
  g_ctx.ring = static_cast<float*>(
      malloc_psram_prefer(sizeof(float) * kRingCapacity * kSampleDim));

  // MQTT window queue (depth 2, holds MqttWindowData* pointers).
  s_mqtt_queue = xQueueCreate(2, sizeof(MqttWindowData*));

  // Small message queue (depth 8, holds MqttSmallMsg values for infer/train).
  s_mqtt_msg_queue = xQueueCreate(8, sizeof(MqttSmallMsg));

  // MQTT payload pool: 2 × 2.4 KB permanently allocated on PSRAM.
  s_wd_pool[0] = static_cast<MqttWindowData*>(malloc_psram_prefer(sizeof(MqttWindowData)));
  s_wd_pool[1] = static_cast<MqttWindowData*>(malloc_psram_prefer(sizeof(MqttWindowData)));

  if (g_ctx.active_weights == nullptr || g_ctx.staging_weights == nullptr ||
      g_ctx.weights_mutex  == nullptr || g_ctx.ml_mutex        == nullptr ||
      g_ctx.sample_mutex   == nullptr ||
      g_ctx.ring           == nullptr || s_mqtt_queue          == nullptr ||
      s_wd_pool[0]         == nullptr || s_wd_pool[1]          == nullptr) {
    ESP_LOGE(TAG, "Pipeline init failed (OOM or queue)");
    return;
  }

  memset(g_ctx.active_weights,  0, weight_bytes);
  memset(g_ctx.staging_weights, 0, weight_bytes);
  memset(g_ctx.ring, 0, sizeof(float) * kRingCapacity * kSampleDim);

  if (g_ctx.active_weights_q7 == nullptr || g_ctx.staging_weights_q7 == nullptr) {
    ESP_LOGW(TAG, "Q7 weights OOM -> using F32 inference");
  } else {
    memset(g_ctx.active_weights_q7,  0, g_ctx.weight_bytes_q7);
    memset(g_ctx.staging_weights_q7, 0, g_ctx.weight_bytes_q7);
  }

  // Optionally load a previously saved model from SD.
  if (cfg.mode == PipelineMode::Training && cfg.load_model_sd) {
    if (sd_load_model(g_ctx.active_weights, g_ctx.staging_weights, g_ctx.weight_count)) {
      g_ctx.model_version    = 1;   // inference_task treats v>0 as "trained"
      g_ctx.weights_preloaded = true;
    }
  }

  ESP_LOGI(TAG, "Pipeline start: weights=%u floats (~%u KB) q7=%u bytes ring=%ux%u preloaded=%d",
           (unsigned)g_ctx.weight_count,
           (unsigned)(g_ctx.weight_count * sizeof(float) / 1024),
           (unsigned)g_ctx.weight_bytes_q7,
           (unsigned)kRingCapacity, (unsigned)kSampleDim,
           (int)g_ctx.weights_preloaded);

  // Core allocation:
  //  CPU0: comm (strict 1 Hz sensor timing), inference, mqtt (network I/O)
  //  CPU1: training (heavy compute, never blocks CPU0)
  // DataCollection mode skips inference and training tasks.
  xTaskCreatePinnedToCore(comm_task, "comm_task", 8192,    &g_ctx,  6, nullptr, 0);
  xTaskCreatePinnedToCore(mqtt_task, "mqtt_task", 8192,    nullptr, 3, nullptr, 0);
  if (cfg.mode == PipelineMode::Training) {
    xTaskCreatePinnedToCore(inference_task, "infer_task", 12288, &g_ctx, 5,
                            &g_ctx.infer_task_handle, 0);
    // AIfES training path uses deep call chains and sizeable stack frames.
    xTaskCreatePinnedToCore(training_task,  "train_task", 98304, &g_ctx, 1, nullptr, 1);
  }
}

void pipeline_notify_sd_ok() {
  s_sd_ok = true;
  s_sd_open_fail_streak = 0;
  // Ensure required directories exist (mkdir is a no-op if already present).
  SD.mkdir("/models");
  SD.mkdir("/datas");
}

}  // namespace app
}  // namespace tinyae
