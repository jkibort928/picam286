#!/usr/bin/env python3
"""Receive MQTT audio commands and play approved songs through the JBL."""

from pathlib import Path
import json
import os
import subprocess

import paho.mqtt.client as mqtt


AUDIO_DIR = Path(__file__).resolve().parent / "audio"
BROKER = os.getenv("MQTT_BROKER", "192.168.1.8")
TOPIC = os.getenv("MQTT_TOPIC", "room1/audio/set")
JBL_SINK = os.getenv("JBL_SINK", "bluez_output.E8_26_CF_07_74_A0.1")
player = None


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker at {BROKER}")
    client.subscribe(TOPIC)
    print(f"Subscribed to {TOPIC}")


def on_message(client, userdata, message):
    global player
    try:
        data = json.loads(message.payload.decode())
        song = Path(data["song"]).name
        volume = max(0, min(100, int(data.get("volume", 50))))
        song_path = AUDIO_DIR / song
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid MQTT message: {error}")
        return

    if not song_path.is_file():
        print(f"Song does not exist: {song_path}")
        return

    if player and player.poll() is None:
        player.terminate()

    subprocess.run(["pactl", "set-default-sink", JBL_SINK], check=False)
    subprocess.run(["pactl", "set-sink-volume", JBL_SINK, f"{volume}%"], check=False)

    # mpg123 is recommended for MP3 playback on the Pi.
    player = subprocess.Popen(["mpg123", "-q", str(song_path)])
    print(f"Playing {song} at {volume}% through the JBL")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_forever()

