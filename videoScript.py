#!/usr/bin/env python3
import argparse
import glob
import os
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.is_connected = self.cap.isOpened()

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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


def get_cascade_path(filename="haarcascade_frontalface_default.xml"):
    """Locates Haar Cascade XML across Debian system paths and pip packages."""
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        p = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(p):
            return p

    candidates = [
        os.path.join("/usr/share/opencv4/haarcascades", filename),
        os.path.join("/usr/share/opencv/haarcascades", filename),
        os.path.join(os.path.dirname(__file__), filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    print(
        f"[-] Warning: {filename} not found. Run: sudo apt install opencv-data"
    )
    return None


cascade_file = get_cascade_path()
face_cascade = cv2.CascadeClassifier(cascade_file) if cascade_file else None


def process_frame(frame):
    """Downsamples frame for rapid face detection to protect Pi 3 CPU."""
    if face_cascade is not None and not face_cascade.empty():
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (w // 2, h // 2))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
        )

        for x, y, fw, fh in faces:
            x1, y1 = x * 2, y * 2
            x2, y2 = (x + fw) * 2, (y + fh) * 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face Detected",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
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


@app.route("/")
def index():
    return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <title>Pi 3 Vision Feed</title>
            <style>
                body { background-color: #121212; color: #eee; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
                img { max-width: 90vw; border: 2px solid #333; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h3>Raspberry Pi Vision Stream</h3>
            <img src="/video_feed" alt="Video Stream">
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
