#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"

#include <Arduino.h>

#include "arduino_sensors.h"

static const char *TAG = "TinyAE";

extern "C" void app_main(void) {
  // ESP-IDF 기본 초기화(스토리지/키밸류 저장 등에서 많이 필요)
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  ESP_LOGI(TAG, "Boot: ESP-IDF(main) + Arduino(component)");
  ESP_LOGI(TAG, "Heap (total):   %u bytes", esp_get_free_heap_size());
  ESP_LOGI(TAG, "Heap (internal):%u bytes", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  ESP_LOGI(TAG, "Heap (PSRAM):   %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Arduino 라이브러리들이 기대하는 기본 환경(타이머/핀/Serial 등)을 준비
  initArduino();

  // PlatformIO Serial Monitor에서 바로 볼 수 있게 Arduino Serial도 함께 시작
  Serial.begin(115200);
  Serial.println();
  Serial.println("Boot: ESP-IDF(main) + Arduino(component)");

  // 센서 라이브러리(Arduino)들은 별도 task로 격리해서 IDF 쪽 아키텍처를 유지
  start_arduino_sensors();

  // app_main은 그대로 리턴해도 되지만, main task에서 주기적으로 상태 로그를 찍어도 좋음
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(5000));
    ESP_LOGI(TAG, "Heap=%u, PSRAM=%u",
             esp_get_free_heap_size(),
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  }
}
