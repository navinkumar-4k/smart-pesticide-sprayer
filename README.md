# IoT-Based Precision Agriculture: Automated Leaf Disease Detection & Selective Pesticide Spraying System

Production-ready, full-stack real-time IoT Web Dashboard and Edge AI Actuator Engine built with the **Google Antigravity Agentic Async Paradigm** and **FastAPI**.

---

## 🌾 System Architecture

```
[ AI-Thinker ESP32-CAM ]
  │
  ├─► Captures 160x120 RGB565 Frame (38,400 bytes)
  │   HTTP POST /api/predict_and_spray (application/octet-stream)
  │
  ▼
[ FastAPI / Google Antigravity Agentic Backend ]
  │
  ├─► 1. Vectorized Bitwise Unpacking:
  │      val = ((byte1 << 8) | byte2)
  │      R = (val >> 11) & 0x1F  --> 8-bit [0..255]
  │      G = (val >> 5)  & 0x3F  --> 8-bit [0..255]
  │      B = val & 0x1F          --> 8-bit [0..255]
  │      Tensor shape: (120, 160, 3) uint8
  │
  ├─► 2. MobileNetV2 Deep Learning Inference:
  │      Classes: Healthy, Early Blight, Late Blight
  │
  ├─► 3. Autonomous Actuation Trigger Logic:
  │      If Disease in ['Early Blight', 'Late Blight'] and Confidence >= 70%:
  │         PUMP STATE -> ON
  │         HTTP GET http://esp32-leaf.local/control?pump=ON
  │         Hold 3.0 seconds calibrated spray pulse
  │         HTTP GET http://esp32-leaf.local/control?pump=OFF
  │         PUMP STATE -> OFF
  │
  ├─► 4. In-Memory Real-Time State Store:
  │      Daily spray counter, timestamp tracking, event logs
  │
  ▼
[ Modern White & Blue Portal Dashboard (HTML5 / Vanilla JS) ]
  ├─► Real-time Canvas Rendering (Client-side bitwise decoding)
  ├─► Animated 5V Pump Spraying status indicator
  ├─► Dynamic Telemetry Cards & Spray History Table
  └─► Manual Hardware Override Controls
```

---

## ⚡ Technical Specifications

| Feature | Specification |
|---|---|
| **Camera Sensor** | Omnivision OV2640 (AI-Thinker ESP32-CAM) |
| **Resolution & Format** | $160 \times 120$ (QQVGA), Raw RGB565 (16-bit per pixel) |
| **Payload Size** | $160 \times 120 \times 2 = 38,400$ bytes binary payload |
| **Actuator Hardware** | IRLZ44N Logic-Level N-Channel MOSFET driving 5V DC Micro Pesticide Pump |
| **mDNS Target** | `http://esp32-leaf.local` (Local network control) |
| **AI Architecture** | MobileNetV2 Transfer Learning (`tomato_disease_model.h5`) |
| **Classes** | Healthy, Early Blight, Late Blight |
| **Spray Trigger Rule** | Confidence $\ge 70\%$ on pathogenic condition |
| **Spray Duration** | $3.0$ seconds calibrated pulse |

---

## 🔌 Hardware Circuit & Wiring Schematic

### IRLZ44N MOSFET Actuator Circuit (5V Pump)
```
   ESP32-CAM GPIO 14 ───[ 220Ω Resistor ]───► Gate (G) of IRLZ44N
                                   │
                               [ 10kΩ Pull-Down ]
                                   │
                                  GND

   5V DC Supply (+) ────────────────────────► (+) 5V Micro Pesticide Pump
                                                 │
                                            [ 1N4007 Flyback Diode ]
                                                 │
   Drain (D) of IRLZ44N ────────────────────► (-) 5V Micro Pesticide Pump

   Source (S) of IRLZ44N ───────────────────► Common System GND
```

---

## 🚀 Quickstart & Local Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
```bash
python main.py
```
*The server will start at `http://localhost:8000`. Navigate to this URL in any browser to access the live dashboard.*

### 3. Run Hardware Telemetry Simulator
In a separate terminal, test the ingestion pipeline with simulated ESP32-CAM leaf payloads:
```bash
# Test Early Blight (triggers pump spray)
python simulate_esp32.py --condition "Early Blight"

# Test Healthy Foliage (keeps pump idle)
python simulate_esp32.py --condition "Healthy"

# Continuous streaming test loop
python simulate_esp32.py --loop
```

---

## 🌐 API Reference

- `POST /api/predict_and_spray`: Ingests 38,400-byte raw RGB565 binary octet-stream from ESP32-CAM.
- `GET /api/status`: Returns current state snapshot (pump status, counts, confidence, latency).
- `GET /api/history`: Returns structured log of all spray actuations and foliage classifications.
- `POST /api/override/pump`: Body `{"action": "ON" | "OFF"}`. Manual hardware pump override.
- `POST /api/override/monitoring`: Body `{"active": true | false}`. Pause/resume autonomous AI loop.
- `POST /api/simulate_sample`: Body `{"condition": "Healthy" | "Early Blight" | "Late Blight"}`.

---

## ☁️ Cloud Deployment (Google Cloud Run / Heroku)

### Procfile
Included for native PaaS execution:
```text
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Google Cloud Run Deployment
```bash
# Build and deploy directly to Cloud Run
gcloud run deploy precision-ag-engine \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
