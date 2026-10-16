
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature2.live_beat_tracker import LiveBeatTracker


AUDIO_PATH = PROJECT_ROOT / "data" / "raw" / "MEMD_audio" / "10.mp3"

SAMPLE_RATE = 22050
CHUNK_SECONDS = 0.25


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    print("Loading test song...")
    audio, _ = librosa.load(
        str(AUDIO_PATH),
        sr=SAMPLE_RATE,
        mono=True,
    )

    chunk_size = int(SAMPLE_RATE * CHUNK_SECONDS)
    tracker = LiveBeatTracker(sample_rate=SAMPLE_RATE)
    print("Tracker class:", type(tracker))
    print("Tracker source:", __import__("inspect").getfile(type(tracker)))
    detected_beats = 0
    chunks_processed = 0
    last_bpm = None
    last_interval = None

    print("\nTesting live beat tracker with audio chunks...")
    print("This replays a file in chunks; it does not use a microphone.\n")

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]

        if len(chunk) == 0:
            continue

        event = tracker.update(chunk)
        chunks_processed += 1

        # Keep the most recent valid BPM and beat interval.
        if event.get("current_bpm") is not None:
            last_bpm = event["current_bpm"]
            last_interval = event.get("beat_interval_seconds")

        if event.get("beat_detected"):
            detected_beats += 1
            print(
                f"Beat #{detected_beats:03d} | "
                f"time={event.get('timestamp')} | "
                f"BPM={event.get('current_bpm')} | "
                f"interval={event.get('beat_interval_seconds')} | "
                f"strength={event.get('onset_strength')}"
            )

        # Simulate real-time pacing. Remove this delay to run faster.
        time.sleep(CHUNK_SECONDS)

    print("\n=== TEST SUMMARY ===")
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f} seconds")
    print(f"Chunks processed: {chunks_processed}")
    print(f"Beat events detected: {detected_beats}")
    print(f"Final estimated BPM: {last_bpm}")
    print(f"Final beat interval: {last_interval}")

    if detected_beats == 0:
        print(
            "\nNo beats detected. Inspect the tracker's threshold "
            "and chunk-processing logic."
        )
    elif last_bpm is None:
        print(
            "\nBeat events were detected, but no valid BPM estimate "
            "was produced. Inspect the tracker's interval calculation."
        )
    else:
        print("\nCompare with the offline reference of approximately 93.96 BPM.")
        print("A non-None estimate alone does not establish timing accuracy.")


if __name__ == "__main__":
    main()
