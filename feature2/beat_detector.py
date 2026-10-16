
from pathlib import Path

import librosa
import numpy as np


def detect_beats(
    audio_path: str | Path,
    *,
    sample_rate: int = 22050,
    hop_length: int = 256,
    start_bpm: float = 120.0,
) -> dict:
    """Detect beats and estimate when the next beat will occur."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    if y.size == 0 or not np.any(y):
        raise ValueError(f"Audio is empty or silent: {audio_path}")

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        units="frames",
        trim=True,
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = np.asarray(beat_times, dtype=float)

    # Estimate the beat interval from detected timestamps.
    intervals = np.diff(beat_times)
    intervals = intervals[(intervals >= 60 / 200) & (intervals <= 60 / 40)]

    if intervals.size:
        beat_interval = float(np.median(intervals))
        confidence = min(1.0, len(intervals) / 8)
        next_beat = float(beat_times[-1] + beat_interval) if beat_times.size else None
    elif 40 <= bpm <= 200:
        beat_interval = 60.0 / bpm
        confidence = 0.25
        next_beat = float(beat_times[-1] + beat_interval) if beat_times.size else None
    else:
        beat_interval = None
        confidence = 0.0
        next_beat = None

    return {
        "audio_file": audio_path.name,
        "duration_seconds": round(len(y) / sr, 3),
        "estimated_bpm": round(bpm, 2),
        "beat_times_seconds": np.round(beat_times, 3).tolist(),
        "beat_count": int(len(beat_times)),
        "estimated_beat_interval_seconds": (
            round(beat_interval, 3) if beat_interval is not None else None
        ),
        "next_beat_after_analysis_seconds": (
            round(next_beat, 3) if next_beat is not None else None
        ),
        "prediction_confidence": round(confidence, 2),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Detect beats and estimate the next beat."
    )
    parser.add_argument("audio", help="Path to an audio file")
    args = parser.parse_args()

    print(json.dumps(detect_beats(args.audio), indent=2))
