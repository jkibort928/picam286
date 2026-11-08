#!/usr/bin/env python3
import argparse
import base64
import glob
import json
import os
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import paho.mqtt.client as mqtt
MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_EVENTS = os.getenv("MQTT_TOPIC_EVENTS", "room1/presence/events")


os.environ["OPENCV_LOG_LEVEL"] = "OFF"

app = Flask(__name__)

class VideoStream:
    def __init__(self, src=None, width=640, height=480):
        self.width = width
        self.height = height
        self.src = src if src is not None else self.detect_camera_source()
        self.cap = None
        self.is_connected = False

        if self.src is not None:
            print(f"[*] Initializing camera on source: {self.src}")
            is_v4l2 = isinstance(self.src, int) or (isinstance(self.src, str) and self.src.startswith("/dev/video"))
            backend = cv2.CAP_V4L2 if is_v4l2 else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.src, backend)
            
            if is_v4l2:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_connected = self.cap.isOpened()
            
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.was_connected = self.is_connected

    def _test_device(self, dev):
        """Attempts to open a device node and grab a single test frame quickly."""
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False
        # Set short buffer to avoid hanging on non-streaming metadata nodes
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        grabbed, _ = cap.read()
        cap.release()
        return bool(grabbed)

    def detect_camera_source(self):
        """
        Scans /dev/video* nodes numerically and returns the first functional capture device.
        Works seamlessly with USB webcams and libcamerify-wrapped CSI cameras.
        """
        video_nodes = sorted(
            glob.glob("/dev/video*"),
            key=lambda x: int(x.replace("/dev/video", "") or 0),
        )

        for dev in video_nodes:
            if self._test_device(dev):
                print(f"[+] Verified active capture device on: {dev}")
                return dev

        return None

    def _generate_placeholder(self, message="NO CAMERA DETECTED"):
        """Generates a synthetic diagnostic frame when hardware capture fails."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        cv2.putText(
            frame,
            message,
            (40, self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Check ribbon cable, USB connection, or permissions",
            (40, self.height // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        return frame

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _attempt_reconnect(self):
        """Attempts to rediscover and reinitialize a camera device."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        new_src = self.detect_camera_source()
        if new_src is not None:
            self.src = new_src
            is_v4l2 = isinstance(self.src, int) or (isinstance(self.src, str) and self.src.startswith("/dev/video"))
            backend = cv2.CAP_V4L2 if is_v4l2 else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.src, backend)
            
            if is_v4l2:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_connected = self.cap.isOpened()
            
            if self.is_connected:
                print(f"[+] Camera reconnected on: {self.src}")
                self.was_connected = True
                return True
        return False

    def _capture_loop(self):
        """Drains buffer continuously; silently polls for hardware if disconnected."""
        last_reconnect_attempt = 0.0

        while self.running:
            if self.is_connected and self.cap:
                grabbed, frame = self.cap.read()
                if grabbed:
                    with self.lock:
                        self.latest_frame = frame
                    continue
                else:
                    self.is_connected = False

            # State transition: Log once when dropping from connected to disconnected
            if self.was_connected:
                print(f"[-] Video feed lost on {self.src}. Polling for device...")
                self.was_connected = False

            # Serve synthetic placeholder frame
            placeholder = self._generate_placeholder()
            with self.lock:
                self.latest_frame = placeholder

            # Rescan quietly every 2 seconds
            now = time.time()
            if now - last_reconnect_attempt > 2.0:
                last_reconnect_attempt = now
                self._attempt_reconnect()

            time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


# Global capture stream instance
stream = None

# ---------------------------------------------------------
# Profile Management & Model Initialization
# ---------------------------------------------------------
PROFILES_FILE = "profiles.json"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Stage 1: Detects face bounding boxes and 5 landmarks used to align the crop
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
# Stage 2: Converts the aligned face crop into a 128D vector for identity matching
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")


class ProfileDatabase:
    """Stores and matches 128D facial embedding vectors."""

    def __init__(self, filepath=PROFILES_FILE):
        self.filepath = filepath
        self.profiles = {}  # { name: [vector_1, vector_2, ...] }
        self.load()

    def _find_canonical_name(self, name):
        """Returns the existing key matching name case-insensitively, or None."""
        lookup = name.strip().lower()
        for existing in self.profiles:
            if existing.lower() == lookup:
                return existing
        return None

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    self.profiles = {
                        name: [np.array(v, dtype=np.float32) for v in vecs]
                        for name, vecs in data.items()
                    }
                print(f"[+] Loaded {len(self.profiles)} profile(s) from {self.filepath}")
            except Exception as e:
                print(f"[-] Error loading profiles: {e}")
                self.profiles = {}

    def save(self):
        data = {
            name: [v.tolist() for v in vecs]
            for name, vecs in self.profiles.items()
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def match(self, feature, threshold=0.36):
        """
        Calculates cosine similarity against all stored embeddings.
        Returns (best_name, confidence_score).
        """
        if not self.profiles:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        # Flatten live feature vector
        feat_norm = feature.flatten()
        norm_val = np.linalg.norm(feat_norm)
        if norm_val > 0:
            feat_norm = feat_norm / norm_val

        for name, vectors in self.profiles.items():
            for v in vectors:
                v_norm = v.flatten()
                v_div = np.linalg.norm(v_norm)
                if v_div > 0:
                    v_norm = v_norm / v_div
                score = float(np.dot(feat_norm, v_norm))
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < threshold:
            return "Unknown", max(0.0, best_score)
        return best_name, best_score

    def add_feature(self, name, feature):
        """Adds a 128D embedding vector to a profile and saves to disk."""
        name = name.strip()
        if not name:
            return False

        feat_norm = feature.flatten().astype(np.float32)
        norm_val = np.linalg.norm(feat_norm)
        if norm_val > 0:
            feat_norm = feat_norm / norm_val

        # Match existing capitalization if already registered; otherwise preserve new casing
        canonical_name = self._find_canonical_name(name) or name

        if canonical_name not in self.profiles:
            self.profiles[canonical_name] = []

        self.profiles[canonical_name].append(feat_norm)
        self.save()
        print(f"[+] Added sample for '{canonical_name}'. Total samples: {len(self.profiles[canonical_name])}")
        return True

    def delete_profile(self, name):
        """Deletes all vector samples associated with a name (case-insensitive)."""
        canonical_name = self._find_canonical_name(name)
        if canonical_name and canonical_name in self.profiles:
            del self.profiles[canonical_name]
            self.save()
            print(f"[-] Deleted profile '{canonical_name}'")
            return True
        return False

    def list_profiles(self):
        """Returns dict of profile names and sample counts."""
        return {name: len(vecs) for name, vecs in self.profiles.items()}


db = ProfileDatabase()


mqtt_client = mqtt.Client()

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[+] Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[-] Failed to connect to MQTT Broker, return code {rc}")

mqtt_client.on_connect = on_mqtt_connect


def publish_event(topic, payload):
    try:
        data = json.dumps(payload)
        res = mqtt_client.publish(topic, data)
        if res.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[-] MQTT Publish failed with code {res.rc}")
    except Exception as e:
        print(f"[-] MQTT publish error: {e}")


# Initialize YuNet Detector (input size dynamically updated per frame)
if not os.path.exists(YUNET_MODEL):
    raise FileNotFoundError(f"Missing {YUNET_MODEL}. Follow README instructions to download it.")
if not os.path.exists(SFACE_MODEL):
    raise FileNotFoundError(f"Missing {SFACE_MODEL}. Follow README instructions to download it.")

yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
    config="",
    input_size=(320, 240),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=5000,
)
yunet_lock = threading.Lock()

sface = cv2.FaceRecognizerSF.create(
    model=SFACE_MODEL,
    config="",
)
sface_lock = threading.Lock()

def get_confidence_color(score, threshold=0.36):
    """
    Returns BGR tuple:
    - Below threshold: Neutral Gray (128, 128, 128)
    - Low match: Red (0, 0, 255)
    - Mid match: Orange (0, 165, 255)
    - High match: Green (0, 255, 0)
    """
    if score < threshold:
        return (128, 128, 128)

    # Normalize score between threshold and 0.85 (effective ceiling)
    norm = np.clip((score - threshold) / (0.85 - threshold), 0.0, 1.0)

    if norm < 0.5:
        # Interpolate Red (0, 0, 255) to Orange (0, 165, 255)
        factor = norm / 0.5
        g = int(165 * factor)
        return (0, g, 255)
    else:
        # Interpolate Orange (0, 165, 255) to Green (0, 255, 0)
        factor = (norm - 0.5) / 0.5
        r = int(255 * (1.0 - factor))
        g = int(165 + (90 * factor))
        return (0, g, r)

class FaceTracker:
    """
    Lightweight centroid tracker that tracks multiple faces across frames
    and caches deep recognition results per individual face.
    """
    def __init__(self, max_disappeared=30, match_distance=60):
        self.next_id = 0
        self.tracks = {}  # { track_id: {"centroid": (cx, cy), "box": box, "disappeared": 0, "name": "Unknown", "score": 0.0, "color": (128,128,128), "last_rec_frame": 0} }
        self.max_disappeared = max_disappeared
        self.match_distance = match_distance
        self.frame_count = 0

    def _dispatch_enter(self, name):
        print(f"[+] Entered: {name}")
        payload = {
            "event": "person_entered",
            "person": name,
            "timestamp": int(time.time()),
        }
        publish_event(MQTT_TOPIC_EVENTS, payload)
    
    def _dispatch_exit(self, name):
        print(f"[-] Left: {name}")
        payload = {
            "event": "person_left",
            "person": name,
            "timestamp": int(time.time()),
        }
        publish_event(MQTT_TOPIC_EVENTS, payload)
        
    def update(self, detected_faces, frame, db, sface):
        self.frame_count += 1

        # If no faces detected, increment disappeared counter for all active tracks
        if len(detected_faces) == 0:
            to_delete = []
            for tid in self.tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    to_delete.append(tid)
            for tid in to_delete:
                track = self.tracks[tid]
                if track.get("announced"):
                    self._dispatch_exit(track["name"])
                del self.tracks[tid]
            return []

        # Compute centroids for newly detected faces
        current_centroids = []
        for face in detected_faces:
            x, y, fw, fh = face[0:4].astype(int)
            cx = int(x + fw / 2)
            cy = int(y + fh / 2)
            current_centroids.append((cx, cy))

        active_track_ids = list(self.tracks.keys())
        assigned_tracks = set()
        assigned_detections = set()

        # Match detections to existing tracks by Euclidean centroid distance
        if active_track_ids:
            for d_idx, (cx, cy) in enumerate(current_centroids):
                best_dist = float("inf")
                best_tid = None

                for tid in active_track_ids:
                    if tid in assigned_tracks:
                        continue
                    tcx, tcy = self.tracks[tid]["centroid"]
                    dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                    if dist < best_dist and dist < self.match_distance:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    assigned_tracks.add(best_tid)
                    assigned_detections.add(d_idx)
                    # Update existing track
                    track = self.tracks[best_tid]
                    track["centroid"] = (cx, cy)
                    track["box"] = detected_faces[d_idx][0:4].astype(int)
                    track["disappeared"] = 0

                    # Recompute SFace only once every 15 frames for this specific person
                    if (self.frame_count - track["last_rec_frame"] >= 15) or (track["name"] == "Unknown" and self.frame_count - track["last_rec_frame"] >= 5):
                        with sface_lock:
                            aligned = sface.alignCrop(frame, detected_faces[d_idx])
                            feat = sface.feature(aligned)
                        name, score = db.match(feat)

                        if track["name"] != name:
                            if track.get("announced") and track["name"] != "Unknown":
                                # Swap occured, ensure exit announcement of old name
                                self._dispatch_exit(track["name"])
                            track["announced"] = False
                            track["match_streak"] = 1 if name != "Unknown" else 0
                        else:
                            if name != "Unknown":
                                track["match_streak"] += 1
                        
                        track["name"] = name
                        track["score"] = score
                        track["color"] = get_confidence_color(score)
                        track["last_rec_frame"] = self.frame_count
                        
                        if not track["announced"] and track["match_streak"] >= 3:
                            track["announced"] = True
                            self._dispatch_enter(name)                 

        # Register brand-new faces
        for d_idx, (cx, cy) in enumerate(current_centroids):
            if d_idx not in assigned_detections:
                with sface_lock:
                    aligned = sface.alignCrop(frame, detected_faces[d_idx])
                    feat = sface.feature(aligned)
                name, score = db.match(feat)
                color = get_confidence_color(score)
                
                self.tracks[self.next_id] = {
                    "centroid": (cx, cy),
                    "box": detected_faces[d_idx][0:4].astype(int),
                    "disappeared": 0,
                    "name": name,
                    "score": score,
                    "color": color,
                    "last_rec_frame": self.frame_count,
                    "match_streak": 1 if name != "Unknown" else 0,
                    "announced": False,
                    "start_time": time.time(),
                }
                self.next_id += 1

        # Age out unassigned tracks
        to_delete = []
        for tid in self.tracks:
            if tid not in assigned_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    to_delete.append(tid)
        for tid in to_delete:
            track = self.tracks[tid]
            if track.get("announced"):
                self._dispatch_exit(track["name"])
            del self.tracks[tid]

        # Return list of active face render targets for current frame
        return [self.tracks[tid] for tid in self.tracks if self.tracks[tid]["disappeared"] == 0]

# 3 second grace period, 70 pixel same person distance
tracker = FaceTracker(max_disappeared=45, match_distance=70)


def process_frame(frame):
    """Detects faces every frame and uses centroid tracking to cache recognition per person."""
    h, w = frame.shape[:2]
    
    with yunet_lock:
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)
        
    valid_faces = []
    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            fw, fh = box[2], box[3]
            confidence = face[-1]
            # Ignore sensor noise and tiny crops
            if fw >= 50 and fh >= 50 and confidence >= 0.85:
                valid_faces.append(face)

    # Update tracker across all valid faces in frame
    active_faces = tracker.update(valid_faces, frame, db, sface)

    # Draw overlays for each tracked individual
    for person in active_faces:
        x, y, fw, fh = person["box"]
        color = person["color"]
        name = person["name"]
        score = person["score"]

        # Ensure box fits in bounds
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        label = f"{name} ({int(score * 100)}%)" if name != "Unknown" else f"Unknown ({int(score * 100)}%)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, max(0, y - 22)), (x + tw + 6, max(22, y)), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 3, max(16, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0) if color[1] > 200 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    dev_label = stream.src if stream and stream.src else "None (Fallback)"
    cv2.putText(
        frame,
        f"Feed: {dev_label}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return frame

def generate_frames():
    """Generator yielding only the newest frame as MJPEG without queue build-up."""
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        processed = process_frame(frame)
        ret, buffer = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        # Cap network transmission to ~25-30 FPS
        time.sleep(0.033)

# Cache for freeze-frame registration
freeze_cache = {
    "frame": None,
    "faces": None,
    "timestamp": 0.0,
    "lock": threading.Lock(),
}


@app.route("/api/freeze", methods=["POST"])
def api_freeze():
    """Captures current frame, runs detection, and returns face thumbnails."""
    frame = stream.read()
    if frame is None:
        return jsonify({"error": "No camera feed available"}), 503

    h, w = frame.shape[:2]

    
    with yunet_lock:
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)

    thumbnails = []
    with freeze_cache["lock"]:
        freeze_cache["frame"] = frame.copy()
        freeze_cache["faces"] = faces
        freeze_cache["timestamp"] = time.time()

    if faces is not None:
        for idx, face in enumerate(faces):
            box = face[0:4].astype(int)
            x, y, fw, fh = box[0], box[1], box[2], box[3]
            confidence = face[-1]

            if fw < 50 or fh < 50 or confidence < 0.85:
                continue

            # Pad crop slightly
            pad_x = int(fw * 0.1)
            pad_y = int(fh * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                ret, buf = cv2.imencode(".jpg", crop)
                if ret:
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    thumbnails.append({"index": idx, "image_b64": b64})

    return jsonify({"faces_found": len(thumbnails), "thumbnails": thumbnails})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Extracts SFace embedding from the frozen face index and associates it with a name."""
    data = request.get_json() or {}
    face_idx = data.get("index")
    name = data.get("name", "").strip()

    if face_idx is None or not name:
        return jsonify({"error": "Missing 'index' or 'name'"}), 400

    with freeze_cache["lock"]:
        frame = freeze_cache["frame"]
        faces = freeze_cache["faces"]

    if frame is None or faces is None or face_idx >= len(faces):
        return jsonify({"error": "Invalid or expired freeze cache"}), 400

    selected_face = faces[face_idx]
    with sface_lock:
        aligned_face = sface.alignCrop(frame, selected_face)
        feature = sface.feature(aligned_face)

    success = db.add_feature(name, feature)
    return jsonify({"success": success, "name": name, "profiles": db.list_profiles()})


@app.route("/api/profiles", methods=["GET"])
def api_get_profiles():
    return jsonify(db.list_profiles())


@app.route("/api/profiles/<name>", methods=["DELETE"])
def api_delete_profile(name):
    success = db.delete_profile(name)
    return jsonify({"success": success, "profiles": db.list_profiles()})

@app.route("/")
def index():
    return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <title>Pi 3 Vision Feed</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { background-color: #121212; color: #eee; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; margin: 0; padding: 15px; }
                h3 { margin-top: 5px; margin-bottom: 12px; }
                .stream-container { margin-bottom: 15px; }
                img.feed { max-width: 90vw; height: auto; border: 2px solid #333; border-radius: 6px; }
                .controls { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; }
                button { background-color: #0284c7; color: white; border: none; padding: 10px 18px; font-size: 14px; font-weight: 600; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #0369a1; }
                button.danger { background-color: #b91c1c; }
                button.danger:hover { background-color: #991b1b; }
                .panel { max-width: 600px; margin: 0 auto 20px auto; background: #1e1e1e; border: 1px solid #2e2e2e; border-radius: 6px; padding: 16px; text-align: left; }
                .panel h4 { margin-top: 0; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }
                .face-grid { display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-start; }
                .face-card { position: relative; background: #262626; border: 1px solid #3a3a3a; border-radius: 6px; padding: 10px; width: 140px; text-align: center; }
                .face-card .close-btn { position: absolute; top: 4px; right: 6px; background: transparent; color: #888; border: none; font-size: 16px; font-weight: bold; cursor: pointer; padding: 0 4px; line-height: 1; }
                .face-card .close-btn:hover { color: #fff; }
                .face-card img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; border: 1px solid #444; margin-top: 6px; }
                .face-card input { width: 90%; margin-top: 8px; margin-bottom: 6px; padding: 5px; background: #121212; border: 1px solid #444; color: #eee; border-radius: 3px; font-size: 12px; }
                .face-card button.save-btn { width: 100%; padding: 6px; font-size: 12px; }
                ul.profile-list { list-style: none; padding: 0; margin: 0; }
                ul.profile-list li { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #2a2a2a; font-size: 14px; }
                ul.profile-list button { padding: 4px 8px; font-size: 11px; }
            </style>
        </head>
        <body>
            <h3>Raspberry Pi Vision Stream</h3>
            
            <div class="stream-container">
                <img id="liveFeed" class="feed" src="/video_feed" alt="Video Stream">
            </div>

            <div class="controls">
                <button id="freezeBtn" onclick="freezeAndEnumerate()">Freeze Frame & Register Faces</button>
            </div>

            <!-- Face Registration Panel -->
            <div id="registrationPanel" class="panel" style="display: none;">
                <h4>
                    <span>Detected Faces in Snapshot</span>
                    <button class="close-btn" style="background:transparent;border:none;color:#888;font-size:13px;cursor:pointer;" onclick="closeRegistrationPanel()">Dismiss All</button>
                </h4>
                <div id="faceGrid" class="face-grid"></div>
            </div>

            <!-- Profiles Database Panel -->
            <div class="panel">
                <h4>Registered Profiles</h4>
                <ul id="profileList" class="profile-list">
                    <li><em>Loading profiles...</em></li>
                </ul>
            </div>

            <script>
                async function loadProfiles() {
                    const res = await fetch('/api/profiles');
                    const profiles = await res.json();
                    const listEl = document.getElementById('profileList');
                    listEl.innerHTML = '';

                    const names = Object.keys(profiles);
                    if (names.length === 0) {
                        listEl.innerHTML = '<li><em>No profiles registered yet.</em></li>';
                        return;
                    }

                    names.forEach(name => {
                        const count = profiles[name];
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <span><strong>${name}</strong> (${count} sample${count > 1 ? 's' : ''})</span>
                            <button class="danger" onclick="deleteProfile('${name}')">Delete</button>
                        `;
                        listEl.appendChild(li);
                    });
                }

                async function deleteProfile(name) {
                    if (!confirm(`Delete profile '${name}'?`)) return;
                    await fetch(`/api/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
                    loadProfiles();
                }

                function closeRegistrationPanel() {
                    document.getElementById('registrationPanel').style.display = 'none';
                    document.getElementById('faceGrid').innerHTML = '';
                }

                function removeCard(cardId) {
                    const card = document.getElementById(cardId);
                    if (card) {
                        card.remove();
                    }
                    const grid = document.getElementById('faceGrid');
                    if (grid.children.length === 0) {
                        closeRegistrationPanel();
                    }
                }

                async function freezeAndEnumerate() {
                    const btn = document.getElementById('freezeBtn');
                    btn.disabled = true;
                    btn.innerText = 'Scanning Frame...';

                    try {
                        const res = await fetch('/api/freeze', { method: 'POST' });
                        const data = await res.json();
                        const panel = document.getElementById('registrationPanel');
                        const grid = document.getElementById('faceGrid');
                        grid.innerHTML = '';

                        if (data.faces_found === 0) {
                            grid.innerHTML = '<p style="color: #999;">No faces detected in this snapshot. Adjust angle/lighting and try again.</p>';
                        } else {
                            data.thumbnails.forEach(t => {
                                const cardId = `faceCard_${t.index}`;
                                const card = document.createElement('div');
                                card.className = 'face-card';
                                card.id = cardId;
                                card.innerHTML = `
                                    <button class="close-btn" title="Dismiss" onclick="removeCard('${cardId}')">&times;</button>
                                    <img src="data:image/jpeg;base64,${t.image_b64}" alt="Face #${t.index}">
                                    <input type="text" id="nameInput_${t.index}" placeholder="Enter Name" onkeydown="if(event.key === 'Enter') registerFace(${t.index}, '${cardId}')">
                                    <button class="save-btn" onclick="registerFace(${t.index}, '${cardId}')">Save</button>
                                `;
                                grid.appendChild(card);
                            });
                        }
                        panel.style.display = 'block';
                    } catch (err) {
                        alert('Failed to capture frame: ' + err);
                    } finally {
                        btn.disabled = false;
                        btn.innerText = 'Freeze Frame & Register Faces';
                    }
                }

                async function registerFace(index, cardId) {
                    const input = document.getElementById(`nameInput_${index}`);
                    const name = input.value.trim();
                    if (!name) {
                        alert('Please enter a name for this face.');
                        return;
                    }

                    const res = await fetch('/api/register', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ index: index, name: name })
                    });
                    const data = await res.json();

                    if (data.success) {
                        removeCard(cardId);
                        loadProfiles();
                    } else {
                        alert('Registration error: ' + (data.error || 'Unknown'));
                    }
                }

                loadProfiles();
            </script>
        </body>
        </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi Vision Appliance")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device path (e.g., /dev/video0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web UI (default: 5000)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the Flask web interface",
    )
    args = parser.parse_args()

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[-] Could not connect to MQTT broker: {e}")

    stream = VideoStream(src=args.device)
    stream.start()

    try:
        if not args.headless:
            print(f"[*] Starting web interface at http://0.0.0.0:{args.port}")
            app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
        else:
            print("[*] Running in headless mode. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping stream...")
    finally:
        print("\n[*] Stopping MQTT client and camera stream...")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        stream.stop()
