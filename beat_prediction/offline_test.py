
import json
import numpy as np
import librosa
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_FILE = PROJECT_ROOT / "audio" / "test_song.wav"


def analyze_music(audio_file):
    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    audio, sample_rate = librosa.load(
        str(audio_file),
        sr=22050,
        mono=True
    )

    tempo, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate,
        hop_length=512,
        units="frames"
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate,
        hop_length=512
    )

    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    next_beat_time = None
    beat_interval = None

    if len(beat_times) >= 2:
        recent_beats = beat_times[-5:]
        intervals = np.diff(recent_beats)

        beat_interval = float(np.median(intervals))

        next_beat_time = (
            float(beat_times[-1]) + beat_interval
        )

    elif len(beat_times) == 1 and tempo_bpm > 0:
        beat_interval = 60.0 / tempo_bpm
        next_beat_time = (
            float(beat_times[-1]) + beat_interval
        )

    rms = librosa.feature.rms(y=audio)[0]
    audio_energy = float(np.mean(rms))

    results = {
        "tempo_bpm": round(tempo_bpm, 2),
        "detected_beats": int(len(beat_times)),
        "last_beat_time": (
            round(float(beat_times[-1]), 3)
            if len(beat_times) else None
        ),
        "next_beat_time": (
            round(next_beat_time, 3)
            if next_beat_time is not None else None
        ),
        "beat_interval": (
            round(beat_interval, 3)
            if beat_interval is not None else None
        ),
        "audio_energy": round(audio_energy, 4)
    }

    return results


if __name__ == "__main__":
    try:
        result = analyze_music(AUDIO_FILE)
        print(json.dumps(result, indent=4))

    except Exception as error:
        print("An error occurred:")
        print(error)