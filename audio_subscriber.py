#!/usr/bin/env python3

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
AUDIO_PROFILES_FILE = ROOT / "audio_profiles.json"

MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv(
    "MQTT_TOPIC_EVENTS",
    "room1/presence/events",
)

JBL_SINK = os.getenv(
    "JBL_SINK",
    "bluez_output.E8_26_CF_07_74_A0.1",
)

song_queue = queue.Queue()
queued_people = set()
present_people = set()

state_lock = threading.Lock()
player = None
current_person = None


def load_audio_profiles():
    with AUDIO_PROFILES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


audio_profiles = load_audio_profiles()


def stop_current_song():
    global player, current_person

    if player is not None and player.poll() is None:
        print(f"Stopping song for {current_person}")
        player.terminate()

        try:
            player.wait(timeout=2)
        except subprocess.TimeoutExpired:
            player.kill()

    player = None
    current_person = None


def play_song(person):
    global player, current_person

    profile = audio_profiles.get(person)

    if profile is None:
        print(f"No audio profile found for {person}")
        return

    song_name = Path(profile["song"]).name
    song_path = AUDIO_DIR / song_name

    if not song_path.is_file():
        print(f"Song does not exist: {song_path}")
        return

    volume = max(0, min(100, int(profile.get("volume", 50))))

    subprocess.run(
        ["pactl", "set-default-sink", JBL_SINK],
        check=False,
    )

    subprocess.run(
        ["pactl", "set-sink-volume", JBL_SINK, f"{volume}%"],
        check=False,
    )

    current_person = person

    print(f"Playing {song_name} for {person} at {volume}%")

    player = subprocess.Popen(
        ["mpg123", "-q", str(song_path)]
    )

    # Wait while the song is playing.
    while player.poll() is None:
        time.sleep(0.1)

    print(f"Finished playing {song_name}")

    player = None
    current_person = None


def audio_worker():
    """Play recognized users in the order they entered."""
    global current_person

    while True:
        person = song_queue.get()

        with state_lock:
            queued_people.discard(person)

            # Do not play a song for someone who already left.
            if person not in present_people:
                song_queue.task_done()
                continue

        try:
            play_song(person)
        finally:
            song_queue.task_done()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print(f"MQTT connection failed with return code {rc}")


def on_message(client, userdata, message):
    try:
        event = json.loads(message.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print("Received invalid MQTT JSON")
        return

    event_type = event.get("event")
    person = event.get("person")

    if not person:
        return

    if event_type == "person_entered":
        with state_lock:
            present_people.add(person)

            # Prevent duplicate songs while the same person remains present.
            if (
                person != current_person
                and person not in queued_people
            ):
                queued_people.add(person)
                song_queue.put(person)
                print(f"Queued audio for {person}")

    elif event_type == "person_left":
        with state_lock:
            present_people.discard(person)

        print(f"{person} left the room")

        # Stop the current person's audio after the five-second grace period.
        if person == current_person:
            stop_current_song()


def main():
    worker = threading.Thread(
        target=audio_worker,
        daemon=True,
    )
    worker.start()

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