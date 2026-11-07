#!/usr/bin/env python3
"""Create Kirthana's reference embedding from 15 camera samples."""

from pathlib import Path
import time

import cv2
import numpy as np
from picamera2 import Picamera2


ROOT = Path(__file__).resolve().parent
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
OUTPUT = ROOT / "embeddings" / "kirthana.npy"
WIDTH, HEIGHT = 320, 240
SAMPLES = 15


camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
)
camera.start()
time.sleep(2)

detector = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL), "", (WIDTH, HEIGHT), 0.6, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
features = []

print("Enrollment started. Look at the camera and slowly move your head.")

try:
    while len(features) < SAMPLES:
        frame = camera.capture_array()
        _, faces = detector.detect(frame)
        if faces is None:
            print("No face detected; adjust your position.")
            time.sleep(0.5)
            continue

        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = recognizer.alignCrop(frame, face)
        features.append(recognizer.feature(aligned))
        print(f"Captured sample {len(features)}/{SAMPLES}")
        time.sleep(0.5)
finally:
    camera.stop()

average = np.mean(np.vstack(features), axis=0, keepdims=True).astype(np.float32)
average /= np.linalg.norm(average)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT, average)
print(f"Saved embedding to {OUTPUT}")

