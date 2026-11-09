#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path

import paho.mqtt.client as mqtt


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
AUDIO_PROFILES_FILE = ROOT / "audio_profiles.json"

# Set MQTT_BROKER to the IP address of the computer running Mosquitto.
MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv(
    "MQTT_TOPIC_EVENTS",
    "room1/presence/events",
)

# Replace this with the JBL sink shown by:
# pactl list short sinks
JBL_SINK = os.getenv(
    "JBL_SINK",
    "bluez_output.E8_26_CF_07_74_A0.1",
)

player = None


def load_audio_profiles():
    """Load each user's song and volume settings."""
    if not AUDIO_PROFILES_FILE.exists():
        print(f"Missing audio profile file: {AUDIO_PROFILES_FILE}")
        return {}

    try:
        with AUDIO_PROFILES_FILE.open("r", encoding="utf-8") as file:
            profiles = json.load(file)

        print(f"Loaded audio profiles for: {', '.join(profiles.keys())}")
        return profiles

    except json.JSONDecodeError as error:
        print(f"Invalid JSON in {AUDIO_PROFILES_FILE}: {error}")
        return {}


audio_profiles = load_audio_profiles()


def stop_current_song():
    """Stop the currently playing song, if one exists."""
    global player

    if player is not None and player.poll() is None:
        print("Stopping the current song.")
        player.terminate()

        try:
            player.wait(timeout=2)
        except subprocess.TimeoutExpired:
            player.kill()

    player = None


def play_user_song(person):
    """Look up and play the song assigned to a recognized user."""
    global player

    profile = audio_profiles.get(person)

    if profile is None:
        print(f"No audio profile found for recognized user: {person}")
        return

    song_name = Path(profile["song"]).name
    song_path = AUDIO_DIR / song_name

    if not song_path.is_file():
        print(f"Song does not exist: {song_path}")
        return

    try:
        volume = int(profile.get("volume", 50))
    except (TypeError, ValueError):
        volume = 50

    volume = max(0, min(100, volume))

    # If another user's greeting is playing, replace it.
    stop_current_song()

    # Select the JBL speaker as the audio output.
    subprocess.run(
        ["pactl", "set-default-sink", JBL_SINK],
        check=False,
    )

    # Set the JBL volume.
    subprocess.run(
        ["pactl", "set-sink-volume", JBL_SINK, f"{volume}%"],
        check=False,
    )

    print(f"Playing {song_name} for {person} at {volume}%.")

    player = subprocess.Popen(
        ["mpg123", "-q", str(song_path)]
    )


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print(f"MQTT connection failed with return code {rc}")


def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except json.JSONDecodeError as error:
        print(f"Invalid MQTT JSON: {error}")
        return

    event_type = payload.get("event")
    person = payload.get("person")

    if not person:
        print("MQTT event did not include a person.")
        return

    print(f"Received event: {event_type} for {person}")

    # Play a greeting only when a person enters.
    if event_type == "person_entered":
        play_user_song(person)

    # The song is allowed to finish after the person leaves.
    elif event_type == "person_left":
        print(f"{person} left the room. No new song will be played.")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        keepalive=60,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()