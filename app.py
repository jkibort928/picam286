#!/usr/bin/env python3
"""Multi-user face recognition, browser UI, and MQTT JBL control."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
EMBEDDING_DIR = ROOT / "embeddings"
PROFILES_FILE = ROOT / "profiles.json"
YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.1.8")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "room1/audio/set")

WIDTH, HEIGHT = 320, 240
ENROLLMENT_SAMPLES = 15
# A stricter threshold prevents an unknown person from triggering audio.
MATCH_THRESHOLD = 0.50
MIN_MATCH_MARGIN = 0.05
REQUIRED_MATCHES = 5
NO_FACE_RESET_FRAMES = 15

app = Flask(__name__)


def safe_filename(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return value or "user"


class ProfileStore:
    """Stores each user's song settings and one or more face vectors."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.profiles: dict[str, dict] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if PROFILES_FILE.exists():
            self.profiles = json.loads(PROFILES_FILE.read_text())

        # Automatically migrate the original one-user setup.
        legacy = EMBEDDING_DIR / "kirthana.npy"
        if not self.profiles and legacy.exists():
            self.profiles = {
                "Kirthana": {
                    "embedding": str(legacy.relative_to(ROOT)),
                    "song": "kirthana_song.mp3",
                    "volume": 20,
                }
            }
            self.save()

        self._load_vectors()

    def _load_vectors(self) -> None:
        self.vectors = {}
        for name, profile in self.profiles.items():
            path = ROOT / profile["embedding"]
            if path.exists():
                vectors = np.load(path).astype(np.float32)
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
                self.vectors[name] = vectors
            else:
                print(f"Warning: missing embedding for {name}: {path}")

    def save(self) -> None:
        PROFILES_FILE.write_text(json.dumps(self.profiles, indent=2) + "\n")

    def add_user(self, name: str, vectors: np.ndarray, song: str, volume: int) -> None:
        filename = safe_filename(name) + ".npy"
        path = EMBEDDING_DIR / filename
        np.save(path, vectors.astype(np.float32))
        with self.lock:
            self.profiles[name] = {
                "embedding": str(path.relative_to(ROOT)),
                "song": Path(song).name,
                "volume": max(0, min(100, int(volume))),
            }
            self.vectors[name] = vectors.astype(np.float32)
            self.save()

    def list_users(self) -> dict:
        with self.lock:
            return {
                name: {
                    "song": profile["song"],
                    "volume": profile["volume"],
                    "ready": name in self.vectors,
                }
                for name, profile in self.profiles.items()
            }

    def delete_user(self, name: str) -> bool:
        with self.lock:
            profile = self.profiles.pop(name, None)
            self.vectors.pop(name, None)
            if profile is None:
                return False
            self.save()

        embedding_path = ROOT / profile["embedding"]
        if embedding_path.exists():
            embedding_path.unlink()
        return True

    def match(self, feature: np.ndarray) -> tuple[str, float, float]:
        """Return best name, best score, and second-best score."""
        with self.lock:
            candidates = list(self.vectors.items())
        if not candidates:
            return "Unknown", 0.0, 0.0

        normalized_feature = feature.flatten()
        normalized_feature /= max(np.linalg.norm(normalized_feature), 1e-12)
        scores = []
        for name, vectors in candidates:
            user_scores = []
            for vector in vectors:
                normalized_vector = vector.flatten()
                normalized_vector /= max(np.linalg.norm(normalized_vector), 1e-12)
                user_scores.append(float(np.dot(normalized_feature, normalized_vector)))
            scores.append((max(user_scores), name))

        scores.sort(reverse=True)
        best_score, best_name = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        if best_score < MATCH_THRESHOLD or best_score - second_score < MIN_MATCH_MARGIN:
            return "Unknown", best_score, second_score
        return best_name, best_score, second_score


class RecognitionDemo:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.status = "Starting camera..."
        self.score = 0.0
        self.running = True
        self.consecutive_matches = 0
        self.no_face_frames = 0
        self.announced_user: str | None = None
        self.enrollment_status = "idle"
        self.enrollment_message = "Enrollment is idle."
        self.store = ProfileStore()

        for path in (YUNET_MODEL, SFACE_MODEL):
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path}")

        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (WIDTH, HEIGHT), 0.6, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")

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

    def _set_state(self, status: str, score: float | None = None) -> None:
        with self.lock:
            self.status = status
            if score is not None:
                self.score = score

    def _publish_audio_event(self, name: str) -> None:
        profile = self.store.profiles[name]
        payload = {
            "person": name,
            "song": profile["song"],
            "volume": profile["volume"],
        }
        result = self.mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published audio event for {name}: {payload}")
            self._set_state(f"{name} recognized — playing through JBL")
        else:
            self._set_state(f"{name} recognized, but MQTT publish failed")

    def _recognize(self, frame: np.ndarray):
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return "No face", 0.0, 0.0, None

        face = max(faces, key=lambda item: item[2] * item[3])
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned)
        name, score, second_score = self.store.match(feature)
        return name, score, second_score, face

    def _camera_loop(self) -> None:
        print("Multi-user recognition demo started.")
        while self.running:
            frame = self.camera.capture_array()
            with self.lock:
                self.latest_frame = frame.copy()

            name, score, second_score, face = self._recognize(frame)
            if face is None:
                self.no_face_frames += 1
                if self.no_face_frames >= NO_FACE_RESET_FRAMES:
                    self.consecutive_matches = 0
                    self.announced_user = None
                self._set_state("Scanning — no face detected", score)
            else:
                self.no_face_frames = 0
                x, y, w, h = face[:4].astype(int)
                is_known = name != "Unknown"
                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name}  {score:.3f}", (max(5, x), max(22, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                if is_known:
                    if name == self.announced_user:
                        self._set_state(f"{name} recognized — event already sent", score)
                    else:
                        self.consecutive_matches += 1
                        if self.consecutive_matches >= REQUIRED_MATCHES:
                            self.announced_user = name
                            self._publish_audio_event(name)
                        else:
                            self._set_state(f"{name} match {self.consecutive_matches}/{REQUIRED_MATCHES}", score)
                else:
                    self.consecutive_matches = 0
                    self.announced_user = None
                    self._set_state(f"Unknown face — best {score:.3f}; next {second_score:.3f}", score)

            cv2.putText(frame, "Raspberry Pi Face Recognition", (8, HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
            time.sleep(0.03)

    def start_enrollment(self, name: str, song: str, volume: int) -> bool:
        with self.lock:
            if self.enrollment_status == "running":
                return False
            self.enrollment_status = "running"
            self.enrollment_message = f"Collecting samples for {name}..."
        threading.Thread(target=self._enrollment_worker,
                         args=(name.strip(), song, volume), daemon=True).start()
        return True

    def _enrollment_worker(self, name: str, song: str, volume: int) -> None:
        samples = []
        try:
            while len(samples) < ENROLLMENT_SAMPLES:
                with self.lock:
                    frame = None if self.latest_frame is None else self.latest_frame.copy()
                if frame is None:
                    time.sleep(0.2)
                    continue

                self.detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = self.detector.detect(frame)
                if faces is None:
                    with self.lock:
                        self.enrollment_message = f"No face detected ({len(samples)}/{ENROLLMENT_SAMPLES})"
                    time.sleep(0.3)
                    continue

                face = max(faces, key=lambda item: item[2] * item[3])
                aligned = self.recognizer.alignCrop(frame, face)
                samples.append(self.recognizer.feature(aligned))
                with self.lock:
                    self.enrollment_message = f"Captured {len(samples)}/{ENROLLMENT_SAMPLES} for {name}"
                time.sleep(0.35)

            self.store.add_user(name, np.vstack(samples), song, volume)
            with self.lock:
                self.enrollment_status = "complete"
                self.enrollment_message = f"Enrolled {name} successfully"
            print(self.enrollment_message)
        except Exception as error:
            with self.lock:
                self.enrollment_status = "error"
                self.enrollment_message = str(error)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def status_json(self):
        with self.lock:
            return {"status": self.status, "score": round(self.score, 3),
                    "enrollment_status": self.enrollment_status,
                    "enrollment_message": self.enrollment_message}

    def stop(self):
        self.running = False
        self.camera.stop()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


demo: RecognitionDemo | None = None

PAGE = """
<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personalized JBL Audio</title>
<style>
body{background:#111827;color:#f9fafb;font-family:Arial;text-align:center;padding:20px}
.card{max-width:760px;margin:auto;background:#1f2937;padding:20px;border-radius:14px}
img{width:min(100%,640px);min-height:240px;border:3px solid #374151;border-radius:10px;background:#111827;object-fit:contain}
#status,#enrollStatus{margin-top:14px;padding:12px;border-radius:8px;background:#374151}
input{padding:9px;margin:4px;width:180px;border-radius:5px;border:0}
button{padding:9px 14px;margin:4px;border:0;border-radius:5px;background:#0284c7;color:white;font-weight:bold}
.small{color:#9ca3af;margin-top:12px}.users{text-align:left;margin:16px auto;max-width:600px}
</style></head><body><div class="card"><h1>Personalized JBL Audio</h1>
<img id="liveFeed" src="/frame.jpg" alt="Live camera feed"><div id="status">Connecting...</div>
<p class="small">The camera recognizes the user, publishes an MQTT event, and plays their song through the JBL speaker.</p>
<hr><h2>Enroll a user</h2>
<input id="name" placeholder="User name"><input id="song" placeholder="Song filename, e.g. alice.mp3">
<input id="volume" type="number" min="0" max="100" value="20" placeholder="Volume">
<button onclick="enroll()">Start enrollment</button><div id="enrollStatus">Enrollment is idle.</div>
<div class="users"><h2>Loaded users</h2><ul id="users"></ul></div></div>
<script>
async function refresh(){let d=await (await fetch('/api/status')).json();
document.getElementById('status').textContent=d.status+' | similarity: '+d.score;
document.getElementById('enrollStatus').textContent=d.enrollment_message||'Enrollment is idle.';
let users=await (await fetch('/api/profiles')).json();
document.getElementById('users').innerHTML=Object.entries(users).map(([n,p])=>`<li>${n}: ${p.song} (${p.volume}%) <button onclick="deleteUser('${encodeURIComponent(n)}')">Delete</button></li>`).join('')||'<li>No users loaded.</li>';}
async function enroll(){let body={name:document.getElementById('name').value,song:document.getElementById('song').value,volume:Number(document.getElementById('volume').value)};
let r=await fetch('/api/enroll',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let d=await r.json();alert(d.message||d.error);}
async function deleteUser(encodedName){let name=decodeURIComponent(encodedName);if(!confirm('Delete '+name+' and re-enroll them?'))return;
let r=await fetch('/api/profiles/'+encodeURIComponent(name),{method:'DELETE'});let d=await r.json();alert(d.message||d.error);refresh();}
setInterval(()=>{document.getElementById('liveFeed').src='/frame.jpg?t='+Date.now()},200);
setInterval(refresh,500);refresh();</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/status")
def status():
    return jsonify(demo.status_json())


@app.route("/api/profiles")
def profiles():
    return jsonify(demo.store.list_users())


@app.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    if demo.store.delete_user(name):
        demo.announced_user = None
        demo.consecutive_matches = 0
        return jsonify({"message": f"Deleted {name}. They can be enrolled again."})
    return jsonify({"error": f"User {name} was not found."}), 404


@app.route("/api/enroll", methods=["POST"])
def enroll():
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()
    song = str(data.get("song", "")).strip()
    try:
        volume = int(data.get("volume", 20))
    except (TypeError, ValueError):
        volume = 20
    if not name or not song:
        return jsonify({"error": "Enter a name and song filename."}), 400
    if demo.start_enrollment(name, song, volume):
        return jsonify({"message": f"Enrollment started for {name}. Look at the camera."})
    return jsonify({"error": "Another enrollment is already running."}), 409


@app.route("/video_feed")
def video_feed():
    def frames():
        while True:
            jpeg = demo.get_jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame.jpg")
def frame_jpg():
    jpeg = demo.get_jpeg()
    if jpeg is None:
        return ("Camera frame is not ready", 503)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    demo = RecognitionDemo()
    try:
        print("Open the UI at http://PI_IP_ADDRESS:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop()

