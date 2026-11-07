#!/usr/bin/env python3
"""Live face-recognition web demo for Raspberry Pi + MQTT JBL playback."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
EMBEDDING_FILE = ROOT / "embeddings" / "kirthana.npy"

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.1.8")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "room1/audio/set")
SONG_NAME = os.getenv("SONG_NAME", "kirthana_song.mp3")
VOLUME = int(os.getenv("VOLUME", "20"))

WIDTH = 320
HEIGHT = 240
COSINE_THRESHOLD = 0.363
REQUIRED_MATCHES = 3
NO_FACE_RESET_FRAMES = 15

app = Flask(__name__)


class RecognitionDemo:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.status = "Starting camera..."
        self.score = 0.0
        self.running = True
        self.consecutive_matches = 0
        self.no_face_frames = 0
        self.already_announced = False

        self._check_files()
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (WIDTH, HEIGHT), 0.6, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        self.known_embedding = np.load(EMBEDDING_FILE).astype(np.float32)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.mqtt_client.loop_start()

        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _check_files() -> None:
        missing = [
            str(path)
            for path in (YUNET_MODEL, SFACE_MODEL, EMBEDDING_FILE)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    def _set_state(self, status: str, score: float | None = None) -> None:
        with self.lock:
            self.status = status
            if score is not None:
                self.score = score

    def _publish_audio_event(self) -> None:
        payload = {
            "person": "Kirthana",
            "song": SONG_NAME,
            "volume": VOLUME,
        }
        result = self.mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published Kirthana audio event: {payload}")
            self._set_state("Kirthana recognized — playing through JBL")
        else:
            print(f"MQTT publish failed with code {result.rc}")
            self._set_state("Recognized Kirthana, but MQTT publish failed")

    def _recognize(self, frame: np.ndarray) -> tuple[str, float, np.ndarray | None]:
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return "No face", 0.0, None

        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = self.recognizer.alignCrop(frame, face)
        live_embedding = self.recognizer.feature(aligned)
        score = float(
            self.recognizer.match(
                self.known_embedding,
                live_embedding,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )
        name = "Kirthana" if score >= COSINE_THRESHOLD else "Unknown"
        return name, score, face

    def _camera_loop(self) -> None:
        print("Recognition demo started.")
        while self.running:
            frame = self.camera.capture_array()
            name, score, face = self._recognize(frame)

            if face is None:
                self.no_face_frames += 1
                if self.no_face_frames >= NO_FACE_RESET_FRAMES:
                    self.consecutive_matches = 0
                    self.already_announced = False
                self._set_state("Scanning — no face detected", score)
            else:
                self.no_face_frames = 0
                x, y, w, h = face[:4].astype(int)
                color = (0, 255, 0) if name == "Kirthana" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name}  {score:.3f}"
                cv2.putText(
                    frame,
                    label,
                    (max(5, x), max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if name == "Kirthana":
                    self.consecutive_matches += 1
                    if self.consecutive_matches >= REQUIRED_MATCHES:
                        if not self.already_announced:
                            self.already_announced = True
                            self._publish_audio_event()
                        else:
                            self._set_state("Kirthana recognized — event already sent", score)
                    else:
                        self._set_state(
                            f"Kirthana match {self.consecutive_matches}/{REQUIRED_MATCHES}",
                            score,
                        )
                else:
                    self.consecutive_matches = 0
                    self.already_announced = False
                    self._set_state("Unknown face", score)

            cv2.putText(
                frame,
                "Raspberry Pi Face Recognition",
                (8, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.latest_jpeg = encoded.tobytes()

            time.sleep(0.03)

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> dict:
        with self.lock:
            return {"status": self.status, "score": round(self.score, 3)}

    def stop(self) -> None:
        self.running = False
        self.camera.stop()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


demo: RecognitionDemo | None = None


PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personalized JBL Audio</title>
  <style>
    body { background:#111827; color:#f9fafb; font-family:Arial,sans-serif; text-align:center; padding:20px; }
    .card { max-width:760px; margin:auto; background:#1f2937; padding:20px; border-radius:14px; }
    img { width:min(100%,640px); border:3px solid #374151; border-radius:10px; }
    #status { margin-top:16px; padding:14px; border-radius:8px; background:#374151; font-size:1.1rem; }
    .small { color:#9ca3af; margin-top:10px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Personalized JBL Audio</h1>
    <img src="/video_feed" alt="Live camera feed">
    <div id="status">Connecting...</div>
        <div class="small">
        The camera recognizes the user, publishes an MQTT event, and plays their song through the JBL speaker.
        </div>  
    </div>
  <script>
    async function refresh() {
      const response = await fetch('/api/status');
      const data = await response.json();
      document.getElementById('status').textContent = data.status + ' | similarity: ' + data.score;
    }
    setInterval(refresh, 500);
    refresh();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/status")
def status():
    return jsonify(demo.get_status())


@app.route("/video_feed")
def video_feed():
    def frames():
        while True:
            jpeg = demo.get_jpeg()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    demo = RecognitionDemo()
    try:
        print("Open the UI at http://PI_IP_ADDRESS:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop()

