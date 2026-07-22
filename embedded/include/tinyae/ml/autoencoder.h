#pragma once

#include <cstdint>

namespace tinyae {
namespace ml {

// Autoencoder spec (60-16-4-16-60)
// 10 sensor features × 6 statistics (mean, std, min, max, slope, curvature) = 60
constexpr uint32_t kAutoencoderInputDim = 60;

enum class AutoencoderOptimizer : uint8_t {
  Adam,
  Sgd,
};

struct AutoencoderTrainConfig {
  uint32_t epochs = 200;
  float learn_rate = 0.01f;
  float early_stopping_target_loss = 0.0005f;

  AutoencoderOptimizer optimizer = AutoencoderOptimizer::Sgd;
  float sgd_momentum = 0.9f;  // Used only for SGD.

  // Logs every N epochs (training still yields every epoch to avoid WDT).
  uint32_t log_every_epochs = 50;
};

// Returns the required number of float parameters in flat_weights.
uint32_t autoencoder_weight_count();

// Trains an autoencoder where target == input (reconstruction).
// - `flat_weights` is updated in-place.
// - `train_data` shape is [datasets, kAutoencoderInputDim].
// - `init_weights=true` runs Glorot init once; subsequent calls should use false to continue training.
int8_t autoencoder_train(float *flat_weights,
                         const float *train_data,
                         uint32_t datasets,
                         const AutoencoderTrainConfig &cfg,
                         bool init_weights,
                         float *train_output_scratch);

// Runs inference (reconstruction). `output_data` shape is [datasets, kAutoencoderInputDim].
int8_t autoencoder_infer(float *flat_weights,
                         const float *input_data,
                         uint32_t datasets,
                         float *output_data);

// Q7 (int8) inference helpers:
// - Quantize F32 weights to Q7 weights using a representative input dataset.
// - Run inference using the quantized Q7 weights (inputs/outputs stay float).
uint32_t autoencoder_weight_bytes_q7();

int8_t autoencoder_quantize_f32_to_q7(const float *flat_weights_f32,
                                      const float *representative_data,
                                      uint32_t datasets,
                                      uint8_t *flat_weights_q7_out);

int8_t autoencoder_infer_q7(const uint8_t *flat_weights_q7,
                            const float *input_data,
                            uint32_t datasets,
                            float *output_data);

// Mean MSE across datasets.
float autoencoder_mean_reconstruction_mse(const float *input_data,
                                          const float *output_data,
                                          uint32_t datasets);

}  // namespace ml
}  // namespace tinyae
