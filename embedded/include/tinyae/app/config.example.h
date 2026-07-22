#pragma once
//
// Template config. Copy this file to `config.h` and fill in your own
// WiFi / broker settings:
//
//     cp config.example.h config.h
//
// `config.h` is git-ignored so real credentials never get committed.
//
#include <cstdint>

namespace tinyae {
namespace app {

constexpr const char* kWifiSsid  = "YOUR_WIFI_SSID";
constexpr const char* kWifiPass  = "YOUR_WIFI_PASSWORD";
constexpr const char* kMqttHost  = "192.168.0.10";  // MQTT broker IP (same LAN as device)
constexpr uint16_t    kMqttPort  = 1883;
constexpr const char* kMqttTopicSensor = "TinyAE/sensor";
constexpr const char* kMqttTopicInfer  = "TinyAE/infer";
constexpr const char* kMqttTopicTrain  = "TinyAE/train";
constexpr const char* kTimezone  = "KST-9";

// SD card SPI pins (LOLIN S3 Pro onboard micro-SD).
constexpr int kSdCs   = 46;
constexpr int kSdMosi = 11;
constexpr int kSdMiso = 13;
constexpr int kSdSck  = 12;

}  // namespace app
}  // namespace tinyae
