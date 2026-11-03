```
# picam286

A lightweight, thermal-aware face recognition and tracking appliance optimized for the Raspberry Pi 3 and Linux development workstations. Uses OpenCV DNN with YuNet (detection) and SFace (128D embeddings), backed by a web management interface for identity registration.

---

## 1. Prerequisites & System Setup

### Raspberry Pi 3 (Raspberry Pi OS / Debian Bookworm)
Do **not** compile OpenCV or install `opencv-python` via `pip` on the Raspberry Pi 3. The board's 1GB RAM will run out of memory (OOM crash) during compilation. Use precompiled system packages:

```bash
sudo apt update
sudo apt install -y python3-opencv opencv-data python3-flask python3-numpy v4l-utils wget
```

### Linux Workstation / Distrobox Development (Fedora Silverblue, Ubuntu, Arch)
On x86_64 development machines, precompiled binary wheels install in seconds without building from source:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python flask numpy
```

### Download Pretrained Models
Download the ONNX models for face detection (YuNet) and feature extraction (SFace) into the project root:

```bash
wget -O face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

wget -O face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

### Configure User Permissions
Ensure your user belongs to the `video` group to access camera hardware:

```bash
sudo usermod -aG video $USER
```
Log out and back in, or run `newgrp video`, for the group permissions to take effect.

---

## 2. Hardware Verification

Verify that connected camera devices are recognized by the kernel:

```bash
# Check V4L2 device nodes
v4l2-ctl --list-devices

# Check CSI ribbon cameras on Raspberry Pi
rpicam-hello --list-cameras
```

- **CSI ribbon cameras** interface via the Broadcom ISP pipeline (`bcm2835-unicam` / `libcamera`).
- **USB webcams** interface via standard UVC drivers (`uvcvideo`).

---

## 3. Usage

The web-based enrollment interface runs by default on port `5000`.

### Raspberry Pi with CSI Ribbon Camera
On Raspberry Pi OS (Bullseye / Bookworm), CSI cameras require the `libcamerify` V4L2 compatibility wrapper to route through the hardware Image Signal Processor (ISP):

```bash
libcamerify python3 videoScript.py
```

### Linux PC / Laptop or USB Webcam on Pi
When running on a development workstation, laptop webcam, or standard USB webcam on the Pi, execute directly with Python:

```bash
python3 videoScript.py
```

- **Web Dashboard**: Open `http://<DEVICE_IP>:5000` in a browser.
- **Important**: Always connect over `http://`, not `https://`. Connecting via HTTPS will reject the plain HTTP socket.

### Advanced Flags

* **Specify device manually:**
  ```bash
  python3 videoScript.py --device /dev/video0
  ```
* **Change web port:**
  ```bash
  python3 videoScript.py --port 8080
  ```
* **Run headless (background service without Flask server):**
  ```bash
  python3 videoScript.py --headless
  ```

---

## 4. Key Features & Pipeline Architecture

* **Multi-Person Centroid Tracking**: Uses a lightweight Euclidean centroid tracker (`FaceTracker`) to track multiple faces simultaneously across frames. Each individual retains their own tracking ID, identity label, and confidence score.
* **Recognition Caching & Thermal Protection**: Deep 128D SFace feature extraction is throttled to run only once every 15 frames per tracked individual (or when a new face enters). Bounding box detection runs continuously, keeping CPU utilization on the Pi 3 between 20–30% and preventing thermal throttling down to 600 MHz.
* **False-Positive & Noise Rejection**: YuNet `score_threshold` is set to `0.85` with a minimum bounding box threshold of 50x50 px to eliminate sensor grain and low-light skin-tone false positives.
* **Web Enrollment & Freeze-Frame Registration**: The web interface allows freezing any live frame to detect, crop, and display candidate faces. Names can be typed and submitted by pressing **Enter** or clicking **Save**.
* **Case-Insensitive Profile Persistence**: Profiles are stored locally in `profiles.json`. Names preserve original display capitalization upon creation (e.g., `"Jack"`), but subsequent registrations and lookups match case-insensitively (`"jack"`, `"JACK"`).
* **Hardware Recovery & Anti-Tearing**: Enforces `MJPG` capture at 15 FPS with driver buffer size 1. If a camera disconnects, the system serves a diagnostic placeholder frame and polls for reconnection every 2 seconds without crashing.
* **Thread Safety**: Model inference calls (`yunet` and `sface`) are protected by mutex locks (`threading.Lock`) to eliminate concurrent DNN forward pass assertion crashes between the MJPEG video stream and API worker threads.

---

## 5. Troubleshooting & Known Issues

### `cv2.error: Layer with requested id=-1 not found`
* **Cause**: Stock Debian 12 (Bookworm) repositories provide OpenCV 4.6.0, which contains an upstream bug in its ONNX parser when handling dynamic input sizing with `face_detection_yunet_2023mar.onnx`.
* **Resolution**:
  * **On PC / Laptop / Distrobox**: Install modern OpenCV wheels in a virtual environment (`pip install --upgrade opencv-python`).
  * **On Raspberry Pi 3**: Download the pinned OpenCV 4.6-compatible 2022 release:
    ```bash
    wget -O face_detection_yunet_2022mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/7e062e54cf5410c09b795ff71b4a255e58498c79/models/face_detection_yunet/face_detection_yunet_2022mar.onnx
    ```
    Then update `YUNET_MODEL = "face_detection_yunet_2022mar.onnx"` in `videoScript.py`.

### `VIDEOIO(V4L2:/dev/video0): select() timeout` on Pi CSI Camera
* **Cause**: On modern Raspberry Pi OS, the CSI ribbon camera operates under `libcamera` and does not provide an ISP-demodulated V4L2 video stream to OpenCV directly.
* **Resolution**: Launch your script prefixed with `libcamerify`:
  ```bash
  libcamerify python3 videoScript.py
  ```

### Frame Tearing or Horizontal Slicing on USB Webcams
* **Cause**: The camera is outputting uncompressed YUYV video, saturating the Raspberry Pi 3's shared USB 2.0 controller.
* **Resolution**: Ensure `self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))` is active in `VideoStream`.

```
