#pragma once

#include <cstdint>

namespace tinyae {
namespace app {

constexpr const char* kWifiSsid  = "Yonsei-IoT-2G";
constexpr const char* kWifiPass  = "yonseiiot209";
constexpr const char* kMqttHost  = "192.168.0.23";  // Mac local broker
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
