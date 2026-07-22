#pragma once

#include <cstdint>

namespace tinyae {
namespace app {

enum class PipelineMode : uint8_t {
  DataCollection,  // Sensor read + SD log + MQTT only; no AE inference or training.
  Training,        // Full pipeline: AE training + inference + anomaly detection.
};

struct PipelineConfig {
  PipelineMode mode          = PipelineMode::Training;
  bool         load_model_sd = true;  // Training only: load /ae_model.bin from SD.
};

// Starts comm/infer/mqtt/train tasks according to cfg.
void start_pipeline(const PipelineConfig& cfg = PipelineConfig{});

// Called by arduino_sensors after a successful SD.begin().
// pipeline.cpp uses this flag instead of SD.cardType() to avoid
// calling into the SD driver after a failed init.
void pipeline_notify_sd_ok();

}  // namespace app
}  // namespace tinyae
