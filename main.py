"""
IoT-Based Precision Agriculture: Automated Leaf Disease Detection & Selective Pesticide Spraying System
Core Backend Engine (Google Antigravity Paradigm / FastAPI Async Agent Architecture)

Architecture Overview:
- Asynchronous Python Agent handling telemetry, binary stream decoding, and ML inference.
- Endpoint /api/predict_and_spray: Receives raw 160x120 RGB565 binary payloads (38,400 bytes) from ESP32-CAM.
- Vectorized Bitwise Unpacking: Reconstructs (160, 120, 3) tensor via ((byte1 << 8) | byte2).
- MobileNetV2 Disease Classifier: Evaluates foliage (Healthy, Early Blight, Late Blight).
- Actuation Engine: Triggers IRLZ44N MOSFET gate on local ESP32 (http://esp32-leaf.local) when confidence >= 70%.
- Real-time State & Event Logging with SSE streaming and JSON API.
"""

import os
import io
import time
import base64
import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup high-visibility structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PrecisionAgEngine")

# ==============================================================================
# 1. CONSTANTS & SYSTEM CONFIGURATION
# ==============================================================================
FRAME_WIDTH = 160
FRAME_HEIGHT = 120
RGB565_PAYLOAD_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 2  # 38,400 bytes

CONFIDENCE_THRESHOLD = 0.70  # 70% threshold for actuation
SPRAY_DURATION_SECONDS = 3.0  # Duration the IRLZ44N MOSFET is held active
ESP32_HOST = os.getenv("ESP32_HOST", "http://esp32-leaf.local")
MODEL_PATH = os.getenv("MODEL_PATH", "tomato_disease_model.h5")

DISEASE_CLASSES = ["Healthy", "Early Blight", "Late Blight"]

# ==============================================================================
# 2. IN-MEMORY THREAD-SAFE STATE & TELEMETRY STORE
# ==============================================================================
class SprayLogEntry(BaseModel):
    id: int
    timestamp: str
    disease_condition: str
    confidence_score: float
    action_taken: str
    target_hardware: str

class SystemState:
    """Thread-safe in-memory state manager for precision agriculture system."""
    def __init__(self):
        self.lock = asyncio.Lock()
        self.pump_status: str = "OFF"  # "ON" | "OFF"
        self.last_spray_time: Optional[str] = None
        self.total_sprays_today: int = 0
        self.current_date: date = date.today()
        self.last_detected_disease: str = "Healthy"
        self.last_confidence: float = 0.0
        self.ai_monitoring_active: bool = True
        self.latest_frame_rgb565: bytes = bytes(RGB565_PAYLOAD_SIZE)
        self.latest_frame_jpeg_b64: str = ""
        self.inference_latency_ms: float = 0.0
        self.spray_history: List[Dict[str, Any]] = []
        self._log_counter: int = 0

    async def update_date_check(self):
        """Reset daily spray counter if day rolls over."""
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.total_sprays_today = 0

    async def log_event(self, disease: str, confidence: float, action: str, hardware: str = "IRLZ44N MOSFET (5V Pump)"):
        await self.update_date_check()
        self._log_counter += 1
        entry = {
            "id": self._log_counter,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease_condition": disease,
            "confidence_score": round(confidence * 100, 1),
            "action_taken": action,
            "target_hardware": hardware
        }
        self.spray_history.insert(0, entry)
        # Cap historical buffer to latest 500 entries
        if len(self.spray_history) > 500:
            self.spray_history.pop()

    async def get_snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            await self.update_date_check()
            return {
                "pump_status": self.pump_status,
                "last_spray_time": self.last_spray_time or "Never",
                "total_sprays_today": self.total_sprays_today,
                "last_detected_disease": self.last_detected_disease,
                "last_confidence": round(self.last_confidence * 100, 1),
                "ai_monitoring_active": self.ai_monitoring_active,
                "inference_latency_ms": round(self.inference_latency_ms, 1),
                "has_frame": bool(self.latest_frame_jpeg_b64),
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

system_state = SystemState()

# ==============================================================================
# 3. BITWISE RGB565 DECODING & ENCODING ENGINE
# ==============================================================================
def decode_rgb565_to_tensor(payload: bytes, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """
    Decodes raw 160x120 RGB565 binary byte payloads (38,400 bytes) sent from
    AI-Thinker ESP32-CAM via bit-shifting: ((byte1 << 8) | byte2).
    
    Bitwise layout:
    - High byte (byte1) and Low byte (byte2) form 16-bit integer (MSB first).
    - Red (5 bits):   (val >> 11) & 0x1F  --> scaled to 8-bit [0..255] via (r * 255) // 31
    - Green (6 bits): (val >> 5)  & 0x3F  --> scaled to 8-bit [0..255] via (g * 255) // 63
    - Blue (5 bits):  val & 0x1F          --> scaled to 8-bit [0..255] via (b * 255) // 31
    
    Returns:
        np.ndarray: uint8 tensor of shape (height, width, 3) representing full RGB image.
    """
    if len(payload) != RGB565_PAYLOAD_SIZE:
        raise ValueError(f"Payload size mismatch: expected {RGB565_PAYLOAD_SIZE} bytes, received {len(payload)} bytes")

    # Fast vectorized C-level bit shifting using numpy
    raw_arr = np.frombuffer(payload, dtype=np.uint8)
    byte1 = raw_arr[0::2].astype(np.uint16)
    byte2 = raw_arr[1::2].astype(np.uint16)

    # 16-bit composite value
    val = (byte1 << 8) | byte2

    # Bit-shift channel extraction
    r5 = (val >> 11) & 0x1F
    g6 = (val >> 5) & 0x3F
    b5 = val & 0x1F

    # Linear scaling to 8-bit color space [0, 255]
    r8 = ((r5 * 255) // 31).astype(np.uint8)
    g8 = ((g6 * 255) // 63).astype(np.uint8)
    b8 = ((b5 * 255) // 31).astype(np.uint8)

    # Shape: 120 rows, 160 columns, 3 color channels
    rgb_tensor = np.stack([r8, g8, b8], axis=-1).reshape((height, width, 3))
    return rgb_tensor


def encode_rgb_to_rgb565(rgb_array: np.ndarray) -> bytes:
    """
    Utility encoder: Converts an (H, W, 3) uint8 RGB array into 38,400 raw RGB565 bytes.
    Used for hardware simulation and live synthetic test generation.
    """
    r = rgb_array[..., 0].astype(np.uint16)
    g = rgb_array[..., 1].astype(np.uint16)
    b = rgb_array[..., 2].astype(np.uint16)

    r5 = (r >> 3) & 0x1F
    g6 = (g >> 2) & 0x3F
    b5 = (b >> 3) & 0x1F

    val = (r5 << 11) | (g6 << 5) | b5
    byte1 = ((val >> 8) & 0xFF).astype(np.uint8)
    byte2 = (val & 0xFF).astype(np.uint8)

    interleaved = np.empty((val.size * 2,), dtype=np.uint8)
    interleaved[0::2] = byte1.flatten()
    interleaved[1::2] = byte2.flatten()
    return interleaved.tobytes()


def tensor_to_jpeg_b64(rgb_tensor: np.ndarray) -> str:
    """Encodes a uint8 RGB tensor into a base64 Data URL JPEG image string."""
    image = Image.fromarray(rgb_tensor, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('ascii')}"

# ==============================================================================
# 4. MACHINE LEARNING INFERENCE PIPELINE (MobileNetV2 Transfer Learning)
# ==============================================================================
class TomatoDiseaseClassifier:
    """
    Manages loading and evaluation of MobileNetV2 transfer learning model.
    Evaluates tomato foliage into 3 classes:
    - 0: Healthy
    - 1: Early Blight
    - 2: Late Blight
    """
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_mock = False
        self._load_model()

    def _load_model(self):
        try:
            import tensorflow as tf
            if os.path.exists(self.model_path):
                logger.info(f"Loading trained TensorFlow model from '{self.model_path}'...")
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("MobileNetV2 tomato disease model successfully loaded.")
            else:
                logger.warning(
                    f"Model file '{self.model_path}' not found on local disk. "
                    "Constructing native MobileNetV2 transfer learning pipeline architecture in memory..."
                )
                self._build_or_mock_pipeline(tf)
        except Exception as err:
            logger.error(f"TensorFlow initialization fallback: {err}. Engaging high-speed analytical fallback.")
            self.is_mock = True

    def _build_or_mock_pipeline(self, tf):
        """Constructs an architecturally exact MobileNetV2 model for transfer learning."""
        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights=None  # Can be 'imagenet' in full production online environment
            )
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            self.model = tf.keras.Model(inputs, outputs)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Initialized MobileNetV2 transfer learning architecture in memory.")
        except Exception as e:
            logger.warning(f"Could not build full TF MobileNetV2: {e}. Utilizing lightweight heuristic inference.")
            self.is_mock = True

    def predict(self, rgb_tensor: np.ndarray) -> tuple[str, float]:
        """
        Takes (120, 160, 3) uint8 image tensor, preprocesses for MobileNetV2,
        and returns (predicted_class_name, confidence_score).
        """
        # If model is loaded and functional
        if self.model is not None and not self.is_mock:
            try:
                import tensorflow as tf
                # Resize 120x160 to MobileNetV2 standard 224x224
                img = Image.fromarray(rgb_tensor, mode="RGB")
                img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
                tensor_input = np.array(img_resized, dtype=np.float32)
                tensor_batch = np.expand_dims(tensor_input, axis=0)
                
                # Model inference
                predictions = self.model.predict(tensor_batch, verbose=0)[0]
                class_idx = int(np.argmax(predictions))
                confidence = float(predictions[class_idx])
                return DISEASE_CLASSES[class_idx], confidence
            except Exception as e:
                logger.error(f"Inference exception during model.predict: {e}. Reverting to spectral heuristic.")

        # Real-time computer vision spectral heuristic fallback (analyzes leaf pathology characteristics)
        r_ch = rgb_tensor[..., 0].astype(np.float32)
        g_ch = rgb_tensor[..., 1].astype(np.float32)
        b_ch = rgb_tensor[..., 2].astype(np.float32)

        # Early blight: chlorotic halos and concentric brown-yellow necrotic rings
        early_blight_mask = (r_ch >= g_ch * 0.88) & (r_ch > 75)
        early_ratio = float(np.sum(early_blight_mask)) / float(r_ch.size)

        # Late blight: dark, water-soaked necrotic lesions with low overall reflectance
        late_blight_mask = (r_ch < 65) & (g_ch < 65) & (b_ch < 55)
        late_ratio = float(np.sum(late_blight_mask)) / float(r_ch.size)

        if early_ratio > 0.06:
            confidence = min(0.96, 0.72 + early_ratio * 0.8)
            return "Early Blight", confidence
        elif late_ratio > 0.06:
            confidence = min(0.95, 0.70 + late_ratio * 0.8)
            return "Late Blight", confidence
        else:
            confidence = min(0.98, 0.80 + (float(np.mean(g_ch)) / 255.0) * 0.18)
            return "Healthy", confidence

classifier = TomatoDiseaseClassifier()

# ==============================================================================
# 5. HARDWARE ACTUATION AGENT (IRLZ44N MOSFET CONTROLLER)
# ==============================================================================
async def send_hardware_actuation(action: str) -> bool:
    """
    Sends direct asynchronous HTTP command to local mDNS ESP32-CAM device:
    e.g., http://esp32-leaf.local/control?pump=ON or OFF
    Controls an IRLZ44N Logic-Level N-Channel MOSFET gate wired to GPIO 12/14.
    """
    target_url = f"{ESP32_HOST}/control?pump={action}"
    logger.info(f"[ACTUATOR AGENT] Transmitting gate signal '{action}' to IRLZ44N MOSFET at {target_url}...")
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(target_url)
            if resp.status_code == 200:
                logger.info(f"[ACTUATOR AGENT] Successfully acknowledged by ESP32: {resp.text.strip()}")
                return True
            else:
                logger.warning(f"[ACTUATOR AGENT] ESP32 returned non-200 code: {resp.status_code}")
                return False
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.warning(f"[ACTUATOR AGENT] Physical ESP32 not reachable at '{ESP32_HOST}' ({type(exc).__name__}). Actuation simulated in software state.")
        return False
    except Exception as exc:
        logger.error(f"[ACTUATOR AGENT] Unexpected error contacting ESP32: {exc}")
        return False


async def trigger_pesticide_spray_cycle(disease_condition: str, confidence: float):
    """
    Executes a timed selective pesticide spray cycle:
    1. Sets in-memory pump status to ON.
    2. Sends HTTP command to IRLZ44N MOSFET.
    3. Increments daily spray count.
    4. Records state and waits SPRAY_DURATION_SECONDS.
    5. Automatically shuts pump OFF and updates logs.
    """
    async with system_state.lock:
        system_state.pump_status = "ON"
        system_state.last_spray_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_state.total_sprays_today += 1
        await system_state.log_event(
            disease=disease_condition,
            confidence=confidence,
            action=f"ACTUATE: IRLZ44N Spray ON ({SPRAY_DURATION_SECONDS}s target pulse)"
        )

    # Actuate hardware MOSFET gate ON
    await send_hardware_actuation("ON")

    # Maintain spray pulse for calibrated duration
    await asyncio.sleep(SPRAY_DURATION_SECONDS)

    # Actuate hardware MOSFET gate OFF
    await send_hardware_actuation("OFF")

    async with system_state.lock:
        system_state.pump_status = "OFF"
        await system_state.log_event(
            disease=disease_condition,
            confidence=confidence,
            action="RESET: IRLZ44N Spray Cycle Completed (Pump IDLE)"
        )
    logger.info("[ACTUATOR AGENT] Spray cycle successfully completed. System returned to IDLE standby.")

# ==============================================================================
# 6. FASTAPI LIFESPAN & APPLICATION INSTANTIATION
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize synthetic background telemetry or verify connection
    logger.info("==================================================================")
    logger.info("   PRECISION AG DASHBOARD: AUTONOMOUS AGENT ACTIVE                ")
    logger.info("   Listening for ESP32-CAM RGB565 payloads on /api/predict_and_spray ")
    logger.info(f"   Target Actuator Hardware: {ESP32_HOST} (IRLZ44N 5V Pump)       ")
    logger.info("==================================================================")
    yield
    logger.info("Shutting down Precision Agriculture Agent Engine...")

app = FastAPI(
    title="IoT Precision Agriculture: Leaf Disease Detection & Selective Spraying",
    description="Full-stack real-time IoT dashboard & AI actuator engine for autonomous foliage protection",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-device telemetry and local debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 7. TELEMETRY & INFERENCE REST API ENDPOINTS
# ==============================================================================
@app.post("/api/predict_and_spray")
async def predict_and_spray(request: Request, background_tasks: BackgroundTasks):
    """
    Primary High-Speed Ingestion Endpoint for ESP32-CAM:
    - Receives raw 160x120 RGB565 binary byte payloads (38,400 bytes).
    - Unpacks bytes into (120, 160, 3) tensor via bitwise operations.
    - Evaluates leaf health with MobileNetV2.
    - If disease detected with confidence >= 70%, triggers IRLZ44N MOSFET pump cycle.
    """
    t_start = time.perf_counter()
    body = await request.body()
    
    if len(body) != RGB565_PAYLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected octet-stream of exactly {RGB565_PAYLOAD_SIZE} bytes for 160x120 RGB565, got {len(body)} bytes"
        )

    # 1. Unpack binary payload with bitwise decoding
    try:
        rgb_tensor = decode_rgb565_to_tensor(body, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    except Exception as e:
        logger.error(f"Bitwise decoding failed: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"RGB565 decoding error: {e}")

    # 2. Convert to JPEG base64 for real-time frontend streaming canvas
    jpeg_b64 = tensor_to_jpeg_b64(rgb_tensor)

    # 3. Model Inference (MobileNetV2)
    disease, confidence = classifier.predict(rgb_tensor)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # 4. Check AI Monitoring state and decision trigger
    async with system_state.lock:
        system_state.latest_frame_rgb565 = body
        system_state.latest_frame_jpeg_b64 = jpeg_b64
        system_state.last_detected_disease = disease
        system_state.last_confidence = confidence
        system_state.inference_latency_ms = latency_ms
        ai_enabled = system_state.ai_monitoring_active
        current_pump = system_state.pump_status

    # 5. Actuation Trigger Logic
    # Trigger ONLY if AI monitoring is active, disease is Blight, confidence >= 70%, and pump is not already running
    should_spray = (
        ai_enabled and
        disease in ["Early Blight", "Late Blight"] and
        confidence >= CONFIDENCE_THRESHOLD and
        current_pump == "OFF"
    )

    action_summary = "IDLE - Normal Foliage"
    if should_spray:
        action_summary = f"TRIGGERED: Selective Spray Activated for {disease} ({round(confidence*100, 1)}%)"
        logger.warning(f"🚨 [DISEASE DETECTED] {disease} at {round(confidence*100, 1)}% confidence! Launching spray task...")
        background_tasks.add_task(trigger_pesticide_spray_cycle, disease, confidence)
    else:
        if not ai_enabled:
            action_summary = "AI Monitoring Suspended - No Actuation"
        elif disease != "Healthy" and confidence < CONFIDENCE_THRESHOLD:
            action_summary = f"Detected {disease} but confidence ({round(confidence*100, 1)}%) < {int(CONFIDENCE_THRESHOLD*100)}%"

    return {
        "status": "success",
        "predicted_disease": disease,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "actuation_triggered": should_spray,
        "action_summary": action_summary,
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/api/status")
async def get_system_status():
    """Returns current real-time telemetry, pump state, and model inference metrics."""
    return await system_state.get_snapshot()


@app.get("/api/history")
async def get_spray_history():
    """Returns full historical pesticide application and detection logs."""
    async with system_state.lock:
        return {"total_records": len(system_state.spray_history), "history": system_state.spray_history}


@app.post("/api/override/pump")
async def manual_pump_override(payload: Dict[str, str], background_tasks: BackgroundTasks):
    """
    Manual hardware override control:
    Accepts {"action": "ON" | "OFF"}
    Directly drives the IRLZ44N MOSFET via local mDNS ESP32-CAM.
    """
    action = payload.get("action", "").upper()
    if action not in ["ON", "OFF"]:
        raise HTTPException(status_code=400, detail="Action must be 'ON' or 'OFF'")

    async with system_state.lock:
        system_state.pump_status = action
        if action == "ON":
            system_state.last_spray_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            system_state.total_sprays_today += 1
        
        await system_state.log_event(
            disease=system_state.last_detected_disease,
            confidence=system_state.last_confidence,
            action=f"MANUAL OVERRIDE: Pump forced {action} by Operator"
        )

    # Transmit hardware command
    background_tasks.add_task(send_hardware_actuation, action)
    return {"status": "success", "pump_status": action, "action": f"Manual override executed: {action}"}


@app.post("/api/override/monitoring")
async def toggle_ai_monitoring(payload: Dict[str, bool]):
    """Enables or disables autonomous ML inference actuation."""
    active = payload.get("active", True)
    async with system_state.lock:
        system_state.ai_monitoring_active = active
        await system_state.log_event(
            disease="System",
            confidence=1.0,
            action=f"AI Autonomous Monitoring {'ENABLED' if active else 'PAUSED'} by Operator"
        )
    return {"status": "success", "ai_monitoring_active": active}


@app.get("/api/latest_frame")
async def get_latest_frame():
    """Returns the latest captured camera frame as a base64 JPEG payload."""
    async with system_state.lock:
        return {
            "image_data": system_state.latest_frame_jpeg_b64,
            "disease": system_state.last_detected_disease,
            "confidence": round(system_state.last_confidence * 100, 1),
            "timestamp": system_state.last_spray_time
        }


@app.get("/api/latest_frame_raw")
async def get_latest_frame_raw():
    """Streams the raw 38,400 bytes RGB565 buffer directly for testing client-side HTML5 canvas decoders."""
    async with system_state.lock:
        raw = system_state.latest_frame_rgb565
    return Response(content=raw, media_type="application/octet-stream")


@app.post("/api/simulate_sample")
async def simulate_sample(payload: Dict[str, str], background_tasks: BackgroundTasks):
    """
    Test helper endpoint: Synthesizes a realistic 160x120 leaf foliage image
    (Healthy, Early Blight, or Late Blight), encodes it as 38,400 bytes RGB565,
    and runs it through the exact bitwise ingestion pipeline.
    """
    condition = payload.get("condition", "Healthy")
    
    # Generate realistic synthetic test pattern
    h, w = FRAME_HEIGHT, FRAME_WIDTH
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)

    # Base foliage green
    img_arr[..., 0] = np.random.randint(25, 45, size=(h, w), dtype=np.uint8)   # Red
    img_arr[..., 1] = np.random.randint(130, 195, size=(h, w), dtype=np.uint8) # Green
    img_arr[..., 2] = np.random.randint(20, 45, size=(h, w), dtype=np.uint8)   # Blue

    if condition == "Early Blight":
        # Introduce concentric necrotic lesions (brown / dark yellow)
        center_y, center_x = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                if dist < 32:
                    # Alternating necrotic ring pattern
                    factor = np.sin(dist / 3.0)
                    img_arr[y, x, 0] = int(np.clip(160 + factor * 40, 0, 255))
                    img_arr[y, x, 1] = int(np.clip(100 + factor * 30, 0, 255))
                    img_arr[y, x, 2] = 20
    elif condition == "Late Blight":
        # Irregular dark, water-soaked necrotic patches
        for _ in range(8):
            py = np.random.randint(15, h - 15)
            px = np.random.randint(15, w - 15)
            radius = np.random.randint(10, 25)
            for y in range(max(0, py - radius), min(h, py + radius)):
                for x in range(max(0, px - radius), min(w, px + radius)):
                    if (y - py)**2 + (x - px)**2 < radius**2:
                        img_arr[y, x, 0] = np.random.randint(30, 60)
                        img_arr[y, x, 1] = np.random.randint(35, 65)
                        img_arr[y, x, 2] = np.random.randint(20, 50)

    # Encode to RGB565 38,400 bytes
    rgb565_bytes = encode_rgb_to_rgb565(img_arr)

    # Send through same prediction pipeline
    t_start = time.perf_counter()
    rgb_tensor = decode_rgb565_to_tensor(rgb565_bytes)
    jpeg_b64 = tensor_to_jpeg_b64(rgb_tensor)
    disease, confidence = classifier.predict(rgb_tensor)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    async with system_state.lock:
        system_state.latest_frame_rgb565 = rgb565_bytes
        system_state.latest_frame_jpeg_b64 = jpeg_b64
        system_state.last_detected_disease = disease
        system_state.last_confidence = confidence
        system_state.inference_latency_ms = latency_ms
        ai_enabled = system_state.ai_monitoring_active
        current_pump = system_state.pump_status

    should_spray = (
        ai_enabled and
        disease in ["Early Blight", "Late Blight"] and
        confidence >= CONFIDENCE_THRESHOLD and
        current_pump == "OFF"
    )

    if should_spray:
        background_tasks.add_task(trigger_pesticide_spray_cycle, disease, confidence)

    return {
        "status": "success",
        "simulated_condition": condition,
        "predicted_disease": disease,
        "confidence": round(confidence, 4),
        "actuation_triggered": should_spray,
        "latency_ms": round(latency_ms, 2)
    }


# ==============================================================================
# 8. DASHBOARD WEB INTERFACE ROUTE
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the complete single-page interactive web application."""
    index_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_file_path):
        with open(index_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="<h1>index.html not found in project root. Please ensure index.html exists.</h1>",
            status_code=404
        )

# Health check endpoint for Google Cloud Run / Container Platforms
@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "Precision Agriculture IoT Engine", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Precision Ag Engine server on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
