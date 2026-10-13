
from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root: the folder containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature2.predict import predict_song
from feature2.beat_detector import detect_beats
from final_predictor import predict_lighting


AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "MEMD_audio"


def to_json_safe(value):
    """Convert common NumPy values into JSON-compatible Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def main():
    print("=== LightHive Audio Pipeline Test ===")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    audio_files = sorted(
        path for path in AUDIO_DIR.iterdir()
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No supported audio files found in {AUDIO_DIR}"
        )

    print(f"\nFound {len(audio_files)} audio files.")
    print("Choose one audio file to test:\n")

    for index, path in enumerate(audio_files[:20], start=1):
        print(f"{index}. {path.name}")

    if len(audio_files) > 20:
        print(f"...and {len(audio_files) - 20} more files.")

    choice = input("\nEnter file number (1-20, or less if fewer files): ").strip()

    try:
        file_number = int(choice)
    except ValueError as exc:
        raise ValueError("Enter a valid file number.") from exc

    if not 1 <= file_number <= min(20, len(audio_files)):
        raise ValueError("Selected file number is out of range.")

    audio_path = audio_files[file_number - 1]
    song_id = audio_path.stem

    print(f"\nSelected: {audio_path.name}")
    print(f"Song ID assumption: {song_id}")

    # Stage 1: use the existing trained emotion model.
    print("\n[1/3] Running emotion prediction...")
    emotion = predict_song(song_id)

    if not isinstance(emotion, dict):
        raise TypeError(
            "predict_song(song_id) should return a dictionary. "
            f"Received: {type(emotion).__name__}"
        )

    if "valence_mean" not in emotion or "arousal_mean" not in emotion:
        raise ValueError(
            "Emotion output must contain 'valence_mean' and "
            "'arousal_mean'. Received keys: "
            f"{list(emotion.keys())}"
        )

    print(
        "Emotion prediction:",
        json.dumps(to_json_safe(emotion), indent=2)
    )

    # Stage 2: run the existing offline beat detector on the chosen file.
    print("\n[2/3] Running offline beat detection...")
    beat = detect_beats(str(audio_path))

    if not isinstance(beat, dict):
        raise TypeError(
            "detect_beats(audio_path) should return a dictionary. "
            f"Received: {type(beat).__name__}"
        )

    print(
        "Beat analysis:",
        json.dumps(to_json_safe(beat), indent=2)
    )

    # Stage 3: combine the real model outputs.
    # Offline beat analysis is not a live beat event, so do not set
    # beat_event={"beat_detected": True} here.
    print("\n[3/3] Combining predictions...")
    prediction = predict_lighting(
        mode="performer",
        emotion=emotion,
        beat=beat,
        gesture=None,
        audience=None,
        beat_event=None,
    )

    result = {
        "audio_file": audio_path.name,
        "emotion_prediction": emotion,
        "offline_beat_analysis": beat,
        "lighting_prediction": prediction,
        "test_type": "offline_audio_integration",
        "live_beat_detection": False,
    }

    output_path = PROJECT_ROOT / "audio_pipeline_result.json"
    output_path.write_text(
        json.dumps(to_json_safe(result), indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL LIGHTING PREDICTION ===")
    print(json.dumps(to_json_safe(prediction), indent=2))
    print(f"\nFull result saved to: {output_path}")
    print("\nSUCCESS: the pipeline completed all three stages.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nPIPELINE TEST FAILED: {error}", file=sys.stderr)
        print(
            "\nCheck the error above before proceeding. "
            "A failure may indicate a song-ID mismatch, missing dependency, "
            "or a difference in the existing function's return format.",
            file=sys.stderr,
        )
        raise
