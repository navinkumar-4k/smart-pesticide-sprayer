/**
 * AI-Thinker ESP32-CAM Firmware
 * Project: IoT-Based Precision Agriculture: Automated Leaf Disease Detection & Selective Pesticide Spraying System
 * 
 * Hardware Connections:
 * - AI-Thinker ESP32-CAM (OV2640 sensor)
 * - IRLZ44N N-Channel Logic-Level MOSFET:
 *     * Gate: Connected to ESP32 GPIO 14 (with 10k pull-down resistor to GND)
 *     * Drain: Connected to 5V Micro Pesticide Pump negative terminal
 *     * Source: Connected to Common Ground (GND)
 *     * Flyback Diode (1N4007 or 1N5819): In parallel with 5V pump motor (Cathode to +5V, Anode to Drain)
 * - 5V DC Power Supply Rail shared between Pump & ESP32-CAM (2A minimum recommended)
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <HTTPClient.h>

// -------------------------------------------------------------
// 1. NETWORK & CLOUD BACKEND CREDENTIALS
// -------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Backend API URL (Replace with your server IP / domain)
const char* BACKEND_INGEST_URL = "http://192.168.1.100:8000/api/predict_and_spray";

// -------------------------------------------------------------
// 2. HARDWARE PIN DEFINITIONS (AI-Thinker ESP32-CAM)
// -------------------------------------------------------------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Actuator Pin: IRLZ44N MOSFET Gate
#define PUMP_MOSFET_PIN   14

// Local Web Server for Direct mDNS Actuation
WebServer server(80);

unsigned long lastCaptureTime = 0;
const unsigned long CAPTURE_INTERVAL_MS = 5000; // Capture frame every 5s

// -------------------------------------------------------------
// 3. HTTP SERVER HANDLERS FOR DIRECT ACTUATION
// -------------------------------------------------------------
void handlePumpControl() {
  if (server.hasArg("pump")) {
    String state = server.arg("pump");
    state.toUpperCase();
    if (state == "ON") {
      digitalWrite(PUMP_MOSFET_PIN, HIGH);
      server.send(200, "text/plain", "PUMP_ON_CONFIRMED");
      Serial.println("[ACTUATOR] IRLZ44N Gate Driven HIGH (5V Pump ON)");
    } else if (state == "OFF") {
      digitalWrite(PUMP_MOSFET_PIN, LOW);
      server.send(200, "text/plain", "PUMP_OFF_CONFIRMED");
      Serial.println("[ACTUATOR] IRLZ44N Gate Driven LOW (5V Pump OFF)");
    } else {
      server.send(400, "text/plain", "INVALID_PARAM");
    }
  } else {
    server.send(400, "text/plain", "MISSING_PUMP_PARAM");
  }
}

void handleStatus() {
  int pumpState = digitalRead(PUMP_MOSFET_PIN);
  String json = "{\"device\":\"esp32-leaf.local\",\"pump\":\"" + String(pumpState ? "ON" : "OFF") + "\"}";
  server.send(200, "application/json", json);
}

// -------------------------------------------------------------
// 4. CAMERA INITIALIZATION (160x120 RGB565)
// -------------------------------------------------------------
bool initCameraRGB565() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;

  // Unpack as raw RGB565 (16-bit per pixel)
  config.pixel_format = PIXFORMAT_RGB565;
  // Resolution: 160 x 120 (QQVGA) = 19,200 pixels = 38,400 bytes
  config.frame_size   = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] Initialization failed with error 0x%x\n", err);
    return false;
  }

  // Adjust sensor settings for outdoor agricultural lighting
  sensor_t *s = esp_camera_sensor_get();
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 2);
  s->set_whitebal(s, 1);

  Serial.println("[CAMERA] OV2640 initialized in QQVGA 160x120 RGB565 mode (38,400 bytes)");
  return true;
}

// -------------------------------------------------------------
// 5. TRANSMIT RAW 38,400 BYTES TO BACKEND
// -------------------------------------------------------------
void captureAndTransmitTelemetry() {
  Serial.println("[STREAM] Capturing foliage frame buffer...");
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[STREAM] Frame buffer capture failed!");
    return;
  }

  if (fb->len != 38400) {
    Serial.printf("[STREAM] Warning: Frame buffer length %u != 38400\n", fb->len);
  }

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(BACKEND_INGEST_URL);
    http.addHeader("Content-Type", "application/octet-stream");

    int httpCode = http.POST(fb->buf, fb->len);
    if (httpCode > 0) {
      String response = http.getString();
      Serial.printf("[STREAM] HTTP POST %d | Response: %s\n", httpCode, response.c_str());
    } else {
      Serial.printf("[STREAM] HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  } else {
    Serial.println("[STREAM] WiFi not connected, skipping transmission.");
  }

  esp_camera_fb_return(fb);
}

// -------------------------------------------------------------
// 6. SETUP & MAIN LOOP
// -------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println("\n--- Precision Agriculture ESP32-CAM Node Booting ---");

  // Actuator initialization
  pinMode(PUMP_MOSFET_PIN, OUTPUT);
  digitalWrite(PUMP_MOSFET_PIN, LOW); // Start with pump safely OFF

  // Camera setup
  if (!initCameraRGB565()) {
    Serial.println("Fatal: Camera configuration failed.");
  }

  // WiFi Connection
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected! IP Address: " + WiFi.localIP().toString());

  // Setup mDNS responder as esp32-leaf.local
  if (MDNS.begin("esp32-leaf")) {
    Serial.println("[mDNS] Responder started at: http://esp32-leaf.local");
  }

  // Configure local actuation endpoints
  server.on("/control", handlePumpControl);
  server.on("/status", handleStatus);
  server.begin();
  Serial.println("[SERVER] Local HTTP Actuator Server listening on port 80");
}

void loop() {
  server.handleClient();

  // Periodic Foliage Ingestion Trigger
  unsigned long now = millis();
  if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = now;
    captureAndTransmitTelemetry();
  }
}
