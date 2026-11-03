# picam286

## 1. Prerequisites & System Setup

Do not compile or install OpenCV or dlib via pip. The Pi 3 will run out of memory (OOM crash) during compilation. Use precompiled system packages from the Debian repositories.

### Install Packages
Run the following commands to install OpenCV, cascade classifier data files, Flask, NumPy, and V4L utilities:

```bash
sudo apt update
sudo apt install -y python3-opencv opencv-data python3-flask python3-numpy v4l-utils
```

## 2. Hardware Verification

Verify that connected camera devices are recognized by the kernel:

```bash
v4l2-ctl --list-devices
```

* CSI ribbon cameras enumerate under drivers such as `unicam` or `bcm2835`.
* USB webcams enumerate under `uvcvideo`.

---

## 3. Usage

### Launch with Web Stream
Start the application with the debug web stream enabled on port 5000:

```bash
python3 videoScript.py --debug-feed
```

* Stream URL: `http://<RASPBERRY_PI_IP>:5000`

### Specify a Device Manually
To bypass automatic hardware detection and force a specific device node:

```bash
python3 videoScript.py --debug-feed --device /dev/video0
```

---

## 4. Expected Behavior & Diagnostics

* Device Selection: Automatically detects and prioritizes CSI cameras over USB webcams.
* Hotplug Support: If no camera is connected, the script displays a dark diagnostic frame ("NO CAMERA DETECTED") and rescans `/dev/video*` every 2 seconds. Plugging in a camera automatically restores the feed without restarting the process.
