#include "tinyae/ml/autoencoder.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"

#include <aifes.h>

namespace {

static const char *TAG = "AIfES_AE";

// 60 -> 16 -> 4 -> 16 -> 60
// Input: 10 features × 6 statistics (mean, std, min, max, slope, curvature).
// Bottleneck of 4 forces compact latent representation of normal behavior.
constexpr uint32_t kLayerCount = 5;

uint32_t s_structure[kLayerCount] = {
    tinyae::ml::kAutoencoderInputDim, 16, 4, 16,
    tinyae::ml::kAutoencoderInputDim};
AIFES_E_activations s_activations[kLayerCount - 1] = {
    AIfES_E_relu, AIfES_E_linear, AIfES_E_relu, AIfES_E_sigmoid};

uint32_t s_epoch_counter = 0;  // Total epochs since boot.
uint32_t s_log_every_epochs = 50;

void loss_callback(float loss) {
  // Training may otherwise starve the idle task (task watchdog), so always yield.
  vTaskDelay(1);

  s_epoch_counter++;
  if (s_log_every_epochs == 0) {
    return;
  }
  if ((s_epoch_counter % s_log_every_epochs) != 0) {
    return;
  }

  const float mse = loss * (float)tinyae::ml::kAutoencoderInputDim;  // AIfES Express reports loss scaled by outputs.
  ESP_LOGI(TAG, "AE epoch(total) %u / loss=%.6f (mse=%.6f)", (unsigned)s_epoch_counter, loss, mse);
}

}  // namespace

namespace tinyae {
namespace ml {

uint32_t autoencoder_weight_count() {
  return AIFES_E_flat_weights_number_fnn_f32(s_structure, kLayerCount);
}

int8_t autoencoder_train(float *flat_weights,
                         const float *train_data,
                         uint32_t datasets,
                         const AutoencoderTrainConfig &cfg,
                         bool init_weights,
                         float *train_output_scratch) {
  if (flat_weights == nullptr || train_data == nullptr || train_output_scratch == nullptr || datasets == 0) {
    return -1;
  }
  if (cfg.learn_rate <= 0.0f) {
    return -1;
  }
  if (cfg.optimizer == AutoencoderOptimizer::Sgd && cfg.sgd_momentum < 0.0f) {
    return -1;
  }

  AIFES_E_model_parameter_fnn_f32 model = {};
  model.layer_count = kLayerCount;
  model.fnn_structure = s_structure;
  model.fnn_activations = s_activations;
  model.flat_weights = flat_weights;

  AIFES_E_init_weights_parameter_fnn_f32 init = {};
  init.init_weights_method = init_weights ? AIfES_E_init_glorot_uniform : AIfES_E_init_no_init;

  AIFES_E_training_parameter_fnn_f32 train = {};
  train.loss = AIfES_E_mse;
  train.learn_rate = cfg.learn_rate;
  if (cfg.optimizer == AutoencoderOptimizer::Sgd) {
    train.optimizer = AIfES_E_sgd;
    train.sgd_momentum = cfg.sgd_momentum;
  } else {
    train.optimizer = AIfES_E_adam;
    train.sgd_momentum = 0.0f;
  }
  train.batch_size = datasets;  // full-batch
  train.epochs = cfg.epochs;

  // Always call the callback every epoch to yield (avoid task_wdt),
  // but throttle actual log output inside loss_callback().
  train.epochs_loss_print_interval = 1;
  train.loss_print_function = loss_callback;
  train.early_stopping = AIfES_E_early_stopping_on;
  train.early_stopping_target_loss = cfg.early_stopping_target_loss;

  s_log_every_epochs = cfg.log_every_epochs;

  const uint16_t in_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};
  const uint16_t out_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};

  // AIfES macros cast away const, but the training does not mutate the input buffer.
  auto *in_ptr = const_cast<float *>(train_data);
  aitensor_t input_tensor = AITENSOR_2D_F32(in_shape, in_ptr);
  aitensor_t target_tensor = AITENSOR_2D_F32(in_shape, in_ptr);

  aitensor_t output_tensor = AITENSOR_2D_F32(out_shape, train_output_scratch);

  return AIFES_E_training_fnn_f32(&input_tensor, &target_tensor, &model, &train, &init, &output_tensor);
}

int8_t autoencoder_infer(float *flat_weights,
                         const float *input_data,
                         uint32_t datasets,
                         float *output_data) {
  if (flat_weights == nullptr || input_data == nullptr || output_data == nullptr || datasets == 0) {
    return -1;
  }

  AIFES_E_model_parameter_fnn_f32 model = {};
  model.layer_count = kLayerCount;
  model.fnn_structure = s_structure;
  model.fnn_activations = s_activations;
  model.flat_weights = flat_weights;

  const uint16_t in_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};
  const uint16_t out_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};

  auto *in_ptr = const_cast<float *>(input_data);
  aitensor_t input_tensor = AITENSOR_2D_F32(in_shape, in_ptr);
  aitensor_t output_tensor = AITENSOR_2D_F32(out_shape, output_data);

  return AIFES_E_inference_fnn_f32(&input_tensor, &model, &output_tensor);
}

uint32_t autoencoder_weight_bytes_q7() {
  return AIFES_E_flat_weights_number_fnn_q7(s_structure, kLayerCount);
}

int8_t autoencoder_quantize_f32_to_q7(const float *flat_weights_f32,
                                      const float *representative_data,
                                      uint32_t datasets,
                                      uint8_t *flat_weights_q7_out) {
  if (flat_weights_f32 == nullptr || representative_data == nullptr || flat_weights_q7_out == nullptr || datasets == 0) {
    return -1;
  }

  AIFES_E_model_parameter_fnn_f32 model = {};
  model.layer_count = kLayerCount;
  model.fnn_structure = s_structure;
  model.fnn_activations = s_activations;
  model.flat_weights = const_cast<float *>(flat_weights_f32);

  const uint16_t in_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};
  auto *in_ptr = const_cast<float *>(representative_data);
  aitensor_t representative_tensor = AITENSOR_2D_F32(in_shape, in_ptr);

  return AIFES_E_quantisation_fnn_f32_to_q7(&representative_tensor, &model, flat_weights_q7_out);
}

int8_t autoencoder_infer_q7(const uint8_t *flat_weights_q7,
                            const float *input_data,
                            uint32_t datasets,
                            float *output_data) {
  if (flat_weights_q7 == nullptr || input_data == nullptr || output_data == nullptr || datasets == 0) {
    return -1;
  }

  AIFES_E_model_parameter_fnn_f32 model = {};
  model.layer_count = kLayerCount;
  model.fnn_structure = s_structure;
  model.fnn_activations = s_activations;
  model.flat_weights = const_cast<uint8_t *>(flat_weights_q7);

  const uint16_t in_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};
  const uint16_t out_shape[2] = {(uint16_t)datasets, (uint16_t)kAutoencoderInputDim};

  auto *in_ptr = const_cast<float *>(input_data);
  aitensor_t input_tensor = AITENSOR_2D_F32(in_shape, in_ptr);
  aitensor_t output_tensor = AITENSOR_2D_F32(out_shape, output_data);

  return AIFES_E_inference_fnn_q7(&input_tensor, &model, &output_tensor);
}

float autoencoder_mean_reconstruction_mse(const float *input_data,
                                          const float *output_data,
                                          uint32_t datasets) {
  if (input_data == nullptr || output_data == nullptr || datasets == 0) {
    return 0.0f;
  }

  float mse_sum = 0.0f;
  for (uint32_t i = 0; i < datasets; i++) {
    float mse = 0.0f;
    for (uint32_t j = 0; j < kAutoencoderInputDim; j++) {
      const float diff = output_data[i * kAutoencoderInputDim + j] - input_data[i * kAutoencoderInputDim + j];
      mse += diff * diff;
    }
    mse_sum += mse / (float)kAutoencoderInputDim;
  }
  return mse_sum / (float)datasets;
}

}  // namespace ml
}  // namespace tinyae
