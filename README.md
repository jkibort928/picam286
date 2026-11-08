# Personalized Face Recognition and JBL Audio

This project uses a Raspberry Pi camera to recognize enrolled users. When a
user is recognized, the system publishes an MQTT event containing that user's
name, song, and volume. The audio subscriber receives the event and plays the
user's song through a connected JBL Bluetooth speaker.

## Project flow

Camera
→ Face detection
→ Face recognition
→ Identify enrolled user
→ Publish MQTT audio event
→ Set volume
→ Play that user's song through the JBL speaker

## Folder structure

```text
pi-jbl-face-demo/
├── app.py                    # Flask UI, camera, recognition, enrollment
├── audio_subscriber.py       # Receives MQTT events and plays songs
├── enroll_kirthana.py        # Optional legacy enrollment script
├── mqtt_contract.md          # MQTT message format
├── requirements.txt
├── .gitignore
├── models/                   # Downloaded model files
├── embeddings/               # Generated face embeddings
├── audio/                    # User MP3 files
└── profiles.json             # User names, songs, and settings
```

## 1. Install Raspberry Pi packages

```bash
sudo apt update

sudo apt install -y \
  python3-picamera2 \
  python3-opencv \
  python3-numpy \
  python3-pip \
  mpg123
```

## 2. Create the required folders

```bash
cd ~/pi-jbl-face-demo
mkdir -p models embeddings audio
```

## 3. Download the face-recognition models

```bash
cd ~/pi-jbl-face-demo/models

wget -O face_detection_yunet_2023mar.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

wget -O face_recognition_sface_2021dec.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

Confirm that the model files exist:

```bash
ls -lh ~/pi-jbl-face-demo/models
```

## 4. Connect the JBL speaker

Turn on the JBL speaker and open Bluetooth control:

```bash
bluetoothctl
```

Inside `bluetoothctl`, run:

```text
power on
agent on
default-agent
scan on
```

Find the JBL speaker's MAC address. Then run:

```text
scan off
pair JBL_MAC_ADDRESS
trust JBL_MAC_ADDRESS
connect JBL_MAC_ADDRESS
exit
```

Confirm that the JBL speaker is connected:

```bash
bluetoothctl info JBL_MAC_ADDRESS
pactl list short sinks
```

## 5. Add songs for users

Songs must be stored on the Raspberry Pi in:

```text
/.../pi-jbl-face-demo/audio/
```

From the Mac, copy a song to the Raspberry Pi:

```bash
scp ~/Downloads/..._audio.mp3 \
...@PI_IP_ADDRESS:/...pi-jbl-face-demo/audio/..._audio.mp3
```

Confirm that the song exists on the Raspberry Pi:

```bash
ls -lh ~/pi-jbl-face-demo/audio
```

The filename entered during enrollment must exactly match the filename in the
`audio` folder.

## 6. Start the MQTT audio subscriber

Open the first Raspberry Pi terminal:

```bash
cd ~/pi-jbl-face-demo
python3 audio_subscriber.py
```

Keep this terminal running.

## 7. Start the camera and web application

Open a second Raspberry Pi terminal:

```bash
cd ~/pi-jbl-face-demo
python3 app.py
```

Open the web interface on the Mac:

```text
http://PI_IP_ADDRESS:5000
```

Replace `PI_IP_ADDRESS` with the Raspberry Pi's actual IP address.

## 8. Enroll a user

Use the enrollment form in the web interface.

Enter:

1. The user's name
2. The exact MP3 filename
3. The desired playback volume
4. Select **Start enrollment**

The user should look at the camera while the application captures face samples.

Repeat the process for every user. For example:

```text
Ki.... → ki....._song.mp3
Ka.... → ka....._audio.mp3
User3    → user3_song.mp3
```

Each user's face embedding and audio settings are saved locally.

## 9. Delete and re-enroll a user

To enroll a user again, click the **Delete** button next to that user in the
web interface.

Then enter the user's information again and select **Start enrollment**.

## 10. Recognition and playback process

When the camera recognizes an enrolled user, the application creates an MQTT
message similar to:

```json
{
  "person": "Ki....",
  "song": "ki....._song.mp3",
  "volume": 50
}
```

The audio subscriber receives the message and:

1. Identifies the user
2. Verifies that the song exists
3. Sets the requested volume
4. Plays the song through the JBL speaker

Unknown faces should not trigger audio playback.

## 11. Manually test MQTT playback

The MQTT audio system can be tested without face recognition:

```bash
mosquitto_pub \
  -h MQTT_BROKER_IP \
  -t "room1/audio/set" \
  -m '{"person":"Ka.....","song":"ka...._audio.mp3","volume":20}'
```

Replace `MQTT_BROKER_IP` with the IP address of the computer running Mosquitto.

## 12. Optional legacy enrollment script

The original single-user enrollment script can be run with:

```bash
cd ~/pi-jbl-face-demo
python3 enroll_kirthana.py
```

For the current multi-user system, enroll users through the web interface
instead.
```