# picam286

A lightweight, resource-efficient face recognition and presence detection appliance for the Raspberry Pi 3 and Linux workstations. Uses OpenCV DNN with YuNet (detection) and SFace (128D embeddings), backed by a web management interface and an MQTT event architecture.

---

## 1. Quick Start

Run the automated setup script to verify dependencies, check group permissions, download required models into `models/`, and diagnose your camera interface:

```bash
chmod +x setup.sh
./setup.sh
```

---

## 2. Prerequisites & System Setup

### Raspberry Pi 3 (Raspberry Pi OS / Debian Bookworm)
Do not build OpenCV or install `opencv-python` via pip on the Raspberry Pi 3. The 1GB RAM ceiling will trigger out-of-memory errors. Install precompiled packages:

```bash
sudo apt update
sudo apt install -y curl v4l-utils python3-opencv python3-flask python3-numpy python3-paho-mqtt libcamera-tools
```

### Linux Workstation / Distrobox
On x86 development environments, use a standard Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python flask numpy paho-mqtt
```

### Manual Pretrained Model Download
If not using `setup.sh`, manually create `models/` and fetch the ONNX models directly from Hugging Face:

```bash
mkdir -p models

# YuNet Face Detection (2023 Mar)
curl -fL -o models/face_detection_yunet_2023mar.onnx \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

# SFace Face Recognition (2021 Dec)
curl -fL -o models/face_recognition_sface_2021dec.onnx \
  https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx
```

---

## 3. Usage

### Raspberry Pi with CSI Ribbon Camera
On Raspberry Pi OS, CSI cameras interface through libcamera and require the `libcamerify` wrapper for V4L2 compatibility:

```bash
libcamerify python3 videoScript.py
```

### Linux PC / Laptop or USB Webcam on Pi
Run directly with Python:

```bash
python3 videoScript.py
```

Open `http://<DEVICE_IP>:5000` in a browser to view the feed or register faces.

### Network Cameras (RTSP / HTTP)
Remote IP cameras (e.g., RTSP security streams, ESP32-CAM) can be passed directly via `--device`:

```bash
python3 videoScript.py --device "rtsp://192.168.1.50:554/stream"
```

### CLI Flags

- `--device /dev/video0`: Explicitly select a camera node or stream URL.
- `--port 8080`: Custom web server port (default: 5000).
- `--headless`: Run presence detection in the terminal without starting the Flask interface.

---

## 4. Web Registration Interface

1. Navigate to `http://<DEVICE_IP>:5000`.
2. Click **Freeze Frame & Register Faces**.
3. Type the user's name on the detected thumbnail and press **Enter** or click **Save**.
4. Profiles are stored in `profiles.json` and matched case-insensitively in real time.

---

## 5. Troubleshooting & Limitations

### `cv2.error: Layer with requested id=-1 not found`
- **Cause**: Stock Debian 12 (Bookworm) packages OpenCV 4.6.0, which contains an upstream bug handling dynamic input sizing in `face_detection_yunet_2023mar.onnx`.
- **Fix**: If on x86, use pip in a venv to install modern dependencies (noted in section 2).
  If you are unable to do that, download the pinned OpenCV 4.6-compatible 2022 release:
  ```bash
  curl -fL -o models/face_detection_yunet_2022mar.onnx \
    https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2022mar.onnx
  ```
  Then set `YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2022mar.onnx")` in `videoScript.py`.

### `VIDEOIO(V4L2:/dev/video0): select() timeout` on CSI Camera
- **Cause**: CSI cameras do not supply standard V4L2 device frames without ISP processing.
- **Fix**: Launch using `libcamerify python3 videoScript.py`.

### Network Stream Disconnections
- Automatic reconnection and polling are currently optimized for local `/dev/video*` nodes. Dropped RTSP/HTTP connections may block OpenCV's capture thread until restarted.
