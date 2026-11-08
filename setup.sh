#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

# Hugging Face direct raw download endpoints
YUNET_URL="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
SFACE_URL="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"

echo "=== picam286: System Environment & Dependency Check ==="

# 1. User Groups & Permissions
NEED_LOGOUT=0
if groups "$USER" | grep -q '\bvideo\b'; then
    echo "[+] User '$USER' is in the 'video' group."
else
    echo "[-] User '$USER' not in 'video' group. Adding..."
    sudo usermod -aG video "$USER"
    NEED_LOGOUT=1
fi

# 2. Package & Binary Checks
MISSING_PKGS=()

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        MISSING_PKGS+=("$1")
    fi
}

check_cmd curl
check_cmd python3
check_cmd v4l2-ctl

# Python module checks
if ! python3 -c "import cv2" &>/dev/null; then
    MISSING_PKGS+=("python3-opencv")
fi

if ! python3 -c "import flask" &>/dev/null; then
    MISSING_PKGS+=("python3-flask")
fi

if ! python3 -c "import paho.mqtt.client" &>/dev/null; then
    MISSING_PKGS+=("python3-paho-mqtt")
fi

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo ""
    echo "[!] Missing tools or modules: ${MISSING_PKGS[*]}"
    if [ -f /etc/debian_version ]; then
        echo "    Install on Raspberry Pi OS / Debian with:"
        echo "    sudo apt update && sudo apt install -y curl v4l-utils python3-opencv python3-flask python3-numpy python3-paho-mqtt"
    else
        echo "    Ensure virtual environment dependencies are installed:"
        echo "    pip install opencv-python flask numpy paho-mqtt"
    fi
    echo ""
else
    echo "[+] System binaries and Python dependencies verified."
fi

# 3. Model Verification & Retrieval
mkdir -p "${MODELS_DIR}"

if [ ! -f "${MODELS_DIR}/face_detection_yunet_2023mar.onnx" ]; then
    echo "[*] Downloading YuNet detection model from Hugging Face..."
    curl -fL --progress-bar -o "${MODELS_DIR}/face_detection_yunet_2023mar.onnx" "${YUNET_URL}"
else
    echo "[+] YuNet ONNX model found in models/."
fi

if [ ! -f "${MODELS_DIR}/face_recognition_sface_2021dec.onnx" ]; then
    echo "[*] Downloading SFace recognition model from Hugging Face..."
    curl -fL --progress-bar -o "${MODELS_DIR}/face_recognition_sface_2021dec.onnx" "${SFACE_URL}"
else
    echo "[+] SFace ONNX model found in models/."
fi

# 4. Camera Diagnostics
echo ""
echo "=== Camera Diagnostics ==="

IS_RPI=0
if [ -f /proc/device-tree/model ] && grep -qi "raspberry pi" /proc/device-tree/model; then
    IS_RPI=1
    echo "[+] Detected hardware: $(tr -d '\0' < /proc/device-tree/model)"
fi

HAS_CSI=0
if command -v rpicam-hello &>/dev/null; then
    if rpicam-hello --list-cameras 2>&1 | grep -q "available"; then
        HAS_CSI=1
        echo "[+] Active CSI Camera Board detected."
    fi
fi

if [ $IS_RPI -eq 1 ] && [ $HAS_CSI -eq 1 ]; then
    echo ""
    echo "==================================================================="
    echo " [!] NOTICE: RASPBERRY PI CSI CAMERA DETECTED"
    echo " Modern Raspberry Pi OS routes CSI cameras through libcamera."
    echo " OpenCV VideoCapture requires the libcamerify V4L2 wrapper."
    echo ""
    echo " Launch videoScript.py with:"
    echo "     libcamerify python3 videoScript.py"
    echo "==================================================================="
else
    echo "[+] Standard V4L2 / USB webcam mode."
    echo " Launch videoScript.py with:"
    echo "     python3 videoScript.py"
fi

if [ $NEED_LOGOUT -eq 1 ]; then
    echo ""
    echo "==================================================================="
    echo " [!] WARNING: Group permissions changed."
    echo " Run 'newgrp video' or log out and back in for permissions to take effect."
    echo "==================================================================="
fi
