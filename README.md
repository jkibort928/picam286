# picam286

## 1. Prerequisites & System Setup

Do not compile or install OpenCV or dlib via pip. The Pi 3 will run out of memory (OOM crash) during compilation. Use precompiled system packages from the Debian repositories.

### Install Packages
Run the following commands to install OpenCV, cascade classifier data files, Flask, NumPy, and V4L utilities:

```bash
sudo apt update
sudo apt install -y python3-opencv opencv-data python3-flask python3-numpy v4l-utils wget
```

### Download Pretrained Models
Download the lightweight ONNX models for face detection (YuNet) and feature extraction (SFace):

```bash
wget -O face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

wget -O face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

### Configure User Permissions
Ensure your user has access to video devices:

```bash
sudo usermod -aG video $USER
```

Log out and back in, or run `newgrp video`, for the permission change to take effect.

---

## 2. Hardware Verification

Verify that connected camera devices are recognized by the kernel:

```bash
v4l2-ctl --list-devices
```

- CSI ribbon cameras enumerate under drivers such as `unicam` or `bcm2835`.
- USB webcams enumerate under `uvcvideo`.

---

## 3. Usage

### Launch with Web Stream
Start the application with the debug web stream enabled on port 5000:

```bash
python3 videoScript.py --debug-feed
```

- Stream URL: `http://<RASPBERRY_PI_IP>:5000`
- Note: Connect using `http://`, not `https://`. Navigating via HTTPS will fail with a 400 Bad Request error.

### Headless Mode
Run the detection pipeline locally without running the web server:

```bash
python3 videoScript.py
```

### Specify a Device Manually
To bypass automatic hardware detection and force a specific device node:

```bash
python3 videoScript.py --debug-feed --device /dev/video0
```

---

## 4. Expected Behavior & Diagnostics

- Device Selection: Automatically detects and prioritizes CSI cameras over USB webcams.
- Hotplug Support: If no camera is connected, the script displays a dark diagnostic frame ("NO CAMERA DETECTED") and rescans `/dev/video*` every 2 seconds. Plugging in a camera automatically restores the feed without restarting the process.
- Face Recognition & Profiles: Recognizes registered faces loaded from `profiles.json`. Boxes display in gray for unknown identities, and scale from red to orange to green as confidence increases.
