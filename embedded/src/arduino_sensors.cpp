#include "arduino_sensors.h"

#include "tinyae/app/config.h"
#include "tinyae/app/pipeline.h"
#include "tinyae/data/sensors.h"

#include "esp_log.h"

#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include <WiFi.h>
#include <Wire.h>
#include <time.h>
#include <cerrno>

static const char* TAG = "ArduinoSensors";

static void connect_wifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  ESP_LOGI(TAG, "WiFi connecting to %s ...", tinyae::app::kWifiSsid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(tinyae::app::kWifiSsid, tinyae::app::kWifiPass);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    if (millis() - t0 > 15000) {
      ESP_LOGW(TAG, "WiFi retry");
      WiFi.disconnect(true);
      delay(200);
      WiFi.begin(tinyae::app::kWifiSsid, tinyae::app::kWifiPass);
      t0 = millis();
    }
  }
  ESP_LOGI(TAG, "WiFi OK, IP=%s", WiFi.localIP().toString().c_str());
}

static void sync_ntp() {
  ESP_LOGI(TAG, "NTP sync...");
  setenv("TZ", tinyae::app::kTimezone, 1);
  tzset();
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  for (int i = 0; i < 80; i++) {
    if (time(nullptr) >= 1700000000) {
      ESP_LOGI(TAG, "NTP OK");
      return;
    }
    delay(100);
  }
  ESP_LOGW(TAG, "NTP not synced (fallback to epoch 0)");
}

static SPIClass s_sd_spi(HSPI);

static void init_sd() {
  // Use a dedicated HSPI bus for SD so that SD.begin() owns its full
  // lifecycle.  The Arduino ESP32 SD library crashes (LoadProhibited in
  // sdcard_uninit → spiEndTransaction) when it tries to tear down an
  // SPI bus it did not create.
  s_sd_spi.begin(tinyae::app::kSdSck, tinyae::app::kSdMiso,
                 tinyae::app::kSdMosi, tinyae::app::kSdCs);

  if (!SD.begin(tinyae::app::kSdCs, s_sd_spi, 4000000)) {
    ESP_LOGW(TAG, "SD card not found or init failed");
    return;
  }

  ESP_LOGI(TAG, "SD OK: %llu MB", SD.cardSize() / (1024ULL * 1024ULL));

  // Verify write path once at boot. Some cards initialize but fail file opens.
  File tf = SD.open("/rw_test.txt", FILE_WRITE);
  if (!tf) {
    ESP_LOGW(TAG, "SD write test failed (errno=%d: %s)", errno, strerror(errno));
    return;
  }
  tf.println("ok");
  tf.close();
  SD.remove("/rw_test.txt");
  ESP_LOGI(TAG, "SD write test OK");

  tinyae::app::pipeline_notify_sd_ok();
}

// ---------------------------------------------------------------------------
// Boot menu: shown over Serial after WiFi+NTP, before pipeline starts.
// Returns a PipelineConfig based on user selection (with timeout fallback).
// ---------------------------------------------------------------------------
static tinyae::app::PipelineConfig boot_menu() {
  tinyae::app::PipelineConfig cfg;

  // Flush any stale bytes that arrived during boot.
  while (Serial.available()) Serial.read();

  Serial.println();
  Serial.println("========================================");
  Serial.println("           TinyAE Boot Menu             ");
  Serial.println("========================================");
  Serial.println("  [1] Data Collection Mode");
  Serial.println("  [2] Training / Inference Mode");
  Serial.println("----------------------------------------");
  Serial.println("  No input in 10 s -> [2] Training");
  Serial.println("========================================");
  Serial.print("Choice: ");

  // Wait up to 10 s for mode selection.
  char mode_ch = '2';
  uint32_t t = millis();
  while (millis() - t < 10000) {
    if (Serial.available()) {
      mode_ch = Serial.read();
      Serial.println(mode_ch);
      break;
    }
    delay(50);
  }

  if (mode_ch == '1') {
    Serial.println(">> Data Collection Mode selected.");
    cfg.mode = tinyae::app::PipelineMode::DataCollection;
    return cfg;
  }

  // Training mode sub-menu.
  Serial.println(">> Training / Inference Mode selected.");
  Serial.println();
  Serial.println("  Load existing model from SD?");
  Serial.println("  [Y] Yes - continue from saved weights");
  Serial.println("  [N] No  - start fresh (Glorot init)");
  Serial.println("  No input in 5 s -> [Y] Load");
  Serial.print("Choice: ");

  char load_ch = 'Y';
  t = millis();
  while (millis() - t < 5000) {
    if (Serial.available()) {
      load_ch = Serial.read();
      Serial.println(load_ch);
      break;
    }
    delay(50);
  }

  cfg.mode = tinyae::app::PipelineMode::Training;
  if (load_ch == 'n' || load_ch == 'N') {
    Serial.println(">> Fresh model (Glorot init).");
    cfg.load_model_sd = false;
  } else {
    Serial.println(">> Loading model from SD (if available).");
    cfg.load_model_sd = true;
  }
  return cfg;
}

void start_arduino_sensors() {
  tinyae::data::sensors_init();
  connect_wifi();
  sync_ntp();
  // init_sd() must come AFTER sync_ntp() so that FAT file timestamps
  // reflect real time instead of 1970-01-01.
  // init_sd();  // TODO: SD card crashes when no card is inserted – re-enable when card is available

  // Show boot menu and start pipeline with selected config.
  const tinyae::app::PipelineConfig cfg = boot_menu();
  tinyae::app::start_pipeline(cfg);
}
