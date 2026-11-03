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
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 15)          # Caps USB packet rate to prevent drops
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Drops laggy backlogged frames at driver level

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.was_connected = self.is_connected

    def _test_device(self, dev):
        """Attempts to open a device node and grab a single test frame."""
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False
        grabbed, _ = cap.read()
        cap.release()
        return grabbed

    def detect_camera_source(self):
        """
        Scans V4L2 device nodes, excludes M2M codec/ISP blocks,
        prioritizes CSI ribbon cameras over USB webcams, and tests reads.
        """
        csi_candidates = []
        usb_candidates = []

        # Sort nodes numerically (/dev/video0 before /dev/video10)
        video_devices = sorted(
            glob.glob("/sys/class/video4linux/video*"),
            key=lambda x: int(os.path.basename(x).replace("video", "") or 0),
        )

        for vdev in video_devices:
            dev_name = os.path.basename(vdev)
            node_path = f"/dev/{dev_name}"
            driver_link = os.path.realpath(os.path.join(vdev, "device/driver"))
            driver_name = os.path.basename(driver_link).lower()

            # Skip hardware encoder/decoder and ISP blocks
            if "codec" in driver_name or "isp" in driver_name:
                continue

            # Check for CSI camera drivers vs USB UVC
            if any(k in driver_name for k in ["unicam", "rp1-cfe"]):
                csi_candidates.append(node_path)
            elif "uvcvideo" in driver_name:
                usb_candidates.append(node_path)

        # 1. Prioritize verified CSI cameras
        for dev in csi_candidates:
            if self._test_device(dev):
                print(f"[+] Verified Pi Camera Module on: {dev}")
                return dev

        # 2. Check verified USB webcams
        for dev in usb_candidates:
            if self._test_device(dev):
                print(f"[+] Verified USB Webcam on: {dev}")
                return dev

        # 3. Fallback scan /dev/video0 through /dev/video4
        for idx in range(5):
            dev = f"/dev/video{idx}"
            if os.path.exists(dev) and self._test_device(dev):
                print(f"[+] Verified fallback capture device on: {dev}")
                return dev

        #print("[-] Warning: No functional video capture source could be verified.")
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
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 15)          # Caps USB packet rate to prevent drops
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Drops laggy backlogged frames at driver level
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
# Stage 1: Detects face bounding boxes and 5 landmarks used to align the crop
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
# Stage 2: Converts the aligned face crop into a 128D vector for identity matching
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"

class ProfileDatabase:
    """Stores and matches 128D facial embedding vectors."""

    def __init__(self, filepath=PROFILES_FILE):
        self.filepath = filepath
        self.profiles = {}  # { name: [vector_1, vector_2, ...] }
        self.load()

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

        if name not in self.profiles:
            self.profiles[name] = []

        self.profiles[name].append(feat_norm)
        self.save()
        print(f"[+] Added sample for '{name}'. Total samples: {len(self.profiles[name])}")
        return True

    def delete_profile(self, name):
        """Deletes all vector samples associated with a name."""
        if name in self.profiles:
            del self.profiles[name]
            self.save()
            print(f"[-] Deleted profile '{name}'")
            return True
        return False

    def list_profiles(self):
        """Returns dict of profile names and sample counts."""
        return {name: len(vecs) for name, vecs in self.profiles.items()}

db = ProfileDatabase()

# Initialize YuNet Detector (input size dynamically updated per frame)
if not os.path.exists(YUNET_MODEL):
    raise FileNotFoundError(f"Missing {YUNET_MODEL}. Follow README instructions to download it.")
if not os.path.exists(SFACE_MODEL):
    raise FileNotFoundError(f"Missing {SFACE_MODEL}. Follow README instructions to download it.")

yunet = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL,
    config="",
    input_size=(320, 240),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

sface = cv2.FaceRecognizerSF.create(
    model=SFACE_MODEL,
    config="",
)

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


def process_frame(frame):
    """Detects faces using YuNet, crops/aligns via SFace, and draws colored overlays."""
    h, w = frame.shape[:2]
    yunet.setInputSize((w, h))

    _, faces = yunet.detect(frame)

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            x, y, fw, fh = box[0], box[1], box[2], box[3]

            # Ensure box remains within frame boundaries
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            # Align face crop and extract 128D embedding
            aligned_face = sface.alignCrop(frame, face)
            feature = sface.feature(aligned_face)

            # Match against profile database
            name, score = db.match(feature)
            color = get_confidence_color(score)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            # Draw label with percentage
            if name == "Unknown":
                label = f"Unknown ({int(score * 100)}%)"
            else:
                label = f"{name} ({int(score * 100)}%)"

            # Background pill for readability
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
                                    <input type="text" id="nameInput_${t.index}" placeholder="Enter Name">
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
    parser = argparse.ArgumentParser(description="Pi Vision Streamer")
    parser.add_argument(
        "--debug-feed",
        action="store_true",
        help="Launch Flask web stream on port 5000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device path (e.g., /dev/video0)",
    )
    args = parser.parse_args()

    stream = VideoStream(src=args.device)
    stream.start()

    try:
        if args.debug_feed:
            print("[*] Starting debug server at http://0.0.0.0:5000")
            app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
        else:
            print("[*] Running headless (pass --debug-feed to view stream on port 5000).")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping stream...")
    finally:
        stream.stop()
