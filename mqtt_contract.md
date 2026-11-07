# MQTT Contract

The face-recognition application publishes to:

```text
room1/audio/set
```

Example payload:

```json
{
  "person": "Kirthana",
  "song": "kirthana_song.mp3",
  "volume": 20
}
```

The audio subscriber validates the song name, sets the JBL as the default audio
sink, sets the volume, and plays the file from the local `audio/` directory.

