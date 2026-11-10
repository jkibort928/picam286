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
    """Load the name-to-song mapping."""
    if not AUDIO_PROFILES_FILE.exists():
        print(f"Missing audio profile file: {AUDIO_PROFILES_FILE}")
        return {}

    try:
        with AUDIO_PROFILES_FILE.open("r", encoding="utf-8") as file:
            profiles = json.load(file)

        print(
            "Loaded audio profiles for: "
            + ", ".join(profiles.keys())
        )
        return profiles

    except json.JSONDecodeError as error:
        print(f"Invalid JSON in {AUDIO_PROFILES_FILE}: {error}")
        return {}


audio_profiles = load_audio_profiles()


def stop_current_song():
    """
    Stop the currently playing song.

    The process is saved in a local variable before stopping so that
    another thread cannot cause a NoneType error.
    """
    global player, current_person

    with state_lock:
        process_to_stop = player
        person_to_stop = current_person

        # Clear the shared state immediately.
        player = None
        current_person = None

    if process_to_stop is not None:
        if process_to_stop.poll() is None:
            print(f"Stopping song for {person_to_stop}")
            process_to_stop.terminate()

            try:
                process_to_stop.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process_to_stop.kill()


def play_song(person):
    """Play the song assigned to a recognized person."""
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

    try:
        volume = int(profile.get("volume", 50))
    except (TypeError, ValueError):
        volume = 50

    volume = max(0, min(100, volume))

    # Route audio to the JBL speaker.
    subprocess.run(
        ["pactl", "set-default-sink", JBL_SINK],
        check=False,
    )

    subprocess.run(
        ["pactl", "set-sink-volume", JBL_SINK, f"{volume}%"],
        check=False,
    )

    print(f"Playing {song_name} for {person} at {volume}%")

    # Store the process in a local variable.
    # This prevents the NoneType crash when another thread stops it.
    process = subprocess.Popen(
        ["mpg123", "-q", str(song_path)]
    )

    with state_lock:
        player = process
        current_person = person

    try:
        # Continue monitoring this specific process.
        while process.poll() is None:
            time.sleep(0.1)

    finally:
        # Only clear the global state if it still refers to this process.
        with state_lock:
            if player is process:
                player = None
                current_person = None

    print(f"Finished playing {song_name}")


def audio_worker():
    """
    Play people in the order they were recognized.

    If a person leaves before their turn, their queued song is skipped.
    """
    while True:
        person = song_queue.get()

        try:
            with state_lock:
                queued_people.discard(person)

                # Do not play audio for someone who already left.
                if person not in present_people:
                    continue

            play_song(person)

        finally:
            song_queue.task_done()


def on_connect(client, userdata, flags, rc):
    """Handle the MQTT connection."""
    if rc == 0:
        print(
            f"Connected to MQTT broker at "
            f"{MQTT_BROKER}:{MQTT_PORT}"
        )

        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")

    else:
        print(f"MQTT connection failed with return code {rc}")


def on_message(client, userdata, message):
    """Handle person_entered and person_left events."""
    try:
        event = json.loads(
            message.payload.decode("utf-8")
        )

    except json.JSONDecodeError as error:
        print(f"Invalid MQTT JSON: {error}")
        return

    event_type = event.get("event")
    person = event.get("person")

    if not person:
        print("MQTT event did not contain a person.")
        return

    print(f"Received {event_type} event for {person}")

    if event_type == "person_entered":
        with state_lock:
            present_people.add(person)

            # Prevent duplicate queue entries for the same person.
            already_current = person == current_person
            already_queued = person in queued_people

            if not already_current and not already_queued:
                queued_people.add(person)
                song_queue.put(person)
                print(f"Queued audio for {person}")

    elif event_type == "person_left":
        with state_lock:
            present_people.discard(person)
            is_current_person = person == current_person

        print(f"{person} left the room")

        # This happens after videoScript's five-second grace period.
        if is_current_person:
            stop_current_song()


def create_mqtt_client():
    """
    Create an MQTT client while avoiding the Paho callback warning
    on newer versions.
    """
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1
        )
    except AttributeError:
        # Compatibility with older Paho installations.
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    return client


def main():
    worker = threading.Thread(
        target=audio_worker,
        daemon=True,
    )
    worker.start()

    client = create_mqtt_client()

    print(
        f"Connecting to MQTT broker at "
        f"{MQTT_BROKER}:{MQTT_PORT}"
    )

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        keepalive=60,
    )

    client.loop_forever()


if __name__ == "__main__":
    main()