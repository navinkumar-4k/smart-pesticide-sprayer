"""
ESP32-CAM Hardware Simulator
Simulates an AI-Thinker ESP32-CAM module capturing 160x120 RGB565 foliage frames
and streaming the raw 38,400 bytes binary octet-stream to the Antigravity Precision Ag Engine.

Usage:
    python simulate_esp32.py --condition "Early Blight" --url "http://localhost:8000/api/predict_and_spray"
"""

import sys
import time
import argparse
import numpy as np
import requests

FRAME_WIDTH = 160
FRAME_HEIGHT = 120
RGB565_PAYLOAD_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 2  # 38,400 bytes


def generate_synthetic_foliage_frame(condition: str = "Healthy") -> bytes:
    """
    Generates a 160x120 RGB leaf image and encodes it into 38,400 raw RGB565 bytes
    using bitwise shifts ((r5 << 11) | (g6 << 5) | b5).
    """
    h, w = FRAME_HEIGHT, FRAME_WIDTH
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Base foliage green
    img[..., 0] = np.random.randint(20, 45, size=(h, w), dtype=np.uint8)   # R
    img[..., 1] = np.random.randint(140, 205, size=(h, w), dtype=np.uint8) # G
    img[..., 2] = np.random.randint(15, 45, size=(h, w), dtype=np.uint8)   # B

    if condition == "Early Blight":
        # Introduce necrotic lesions (concentric brown rings)
        center_y, center_x = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                dist = np.hypot(y - center_y, x - center_x)
                if dist < 28:
                    factor = np.sin(dist / 2.5)
                    img[y, x, 0] = int(np.clip(165 + factor * 45, 0, 255))
                    img[y, x, 1] = int(np.clip(95 + factor * 35, 0, 255))
                    img[y, x, 2] = 20
    elif condition == "Late Blight":
        # Dark, irregular water-soaked lesions
        for _ in range(6):
            cy = np.random.randint(15, h - 15)
            cx = np.random.randint(15, w - 15)
            rad = np.random.randint(8, 22)
            for y in range(max(0, cy - rad), min(h, cy + rad)):
                for x in range(max(0, cx - rad), min(w, cx + rad)):
                    if (y - cy)**2 + (x - cx)**2 < rad**2:
                        img[y, x, 0] = np.random.randint(25, 55)
                        img[y, x, 1] = np.random.randint(30, 60)
                        img[y, x, 2] = np.random.randint(15, 45)

    # Bitwise RGB565 packing
    r = img[..., 0].astype(np.uint16)
    g = img[..., 1].astype(np.uint16)
    b = img[..., 2].astype(np.uint16)

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


def main():
    parser = argparse.ArgumentParser(description="Simulate AI-Thinker ESP32-CAM RGB565 Payload Ingestion")
    parser.add_argument("--url", default="http://localhost:8000/api/predict_and_spray", help="Backend ingestion endpoint")
    parser.add_argument("--condition", choices=["Healthy", "Early Blight", "Late Blight"], default="Early Blight", help="Foliage condition to simulate")
    parser.add_argument("--loop", action="store_true", help="Continuously send frames every 3 seconds")
    args = parser.parse_args()

    print("==================================================================")
    print("   AI-Thinker ESP32-CAM Hardware Simulator (RGB565 QQVGA)          ")
    print(f"   Target URL:  {args.url}")
    print(f"   Payload:     {RGB565_PAYLOAD_SIZE} bytes (160x120 x 2 bytes)")
    print(f"   Condition:   {args.condition}")
    print("==================================================================")

    def send_one(cond):
        payload = generate_synthetic_foliage_frame(cond)
        headers = {"Content-Type": "application/octet-stream"}
        t0 = time.perf_counter()
        try:
            res = requests.post(args.url, data=payload, headers=headers, timeout=5.0)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[{time.strftime('%X')}] Status: {res.status_code} | Latency: {elapsed:.1f}ms | Response: {res.text}")
        except Exception as e:
            print(f"[{time.strftime('%X')}] Transmission error: {e}")

    if args.loop:
        conditions = ["Healthy", "Early Blight", "Late Blight"]
        idx = 0
        while True:
            cond = conditions[idx % len(conditions)]
            print(f"\n--- Transmitting frame sample with condition: '{cond}' ---")
            send_one(cond)
            idx += 1
            time.sleep(4.0)
    else:
        send_one(args.condition)


if __name__ == "__main__":
    main()
