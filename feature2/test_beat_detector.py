
from pathlib import Path
import statistics

from feature2.beat_detector import detect_beats

AUDIO_DIR = Path("data/raw/MEMD_audio")


def main():
    files = sorted(AUDIO_DIR.glob("*.mp3"))[:10]

    if not files:
        print(f"No MP3 files found in {AUDIO_DIR.resolve()}")
        return

    for path in files:
        try:
            result = detect_beats(path)
            times = result["beat_times_seconds"]

            intervals = [
                b - a for a, b in zip(times, times[1:])
                if b > a
            ]

            median_interval = (
                statistics.median(intervals)
                if intervals else 0.0
            )

            interval_bpm = (
                60.0 / median_interval
                if median_interval > 0 else 0.0
            )

            print(
                f'{path.name:12} '
                f'BPM={result["estimated_bpm"]:7.2f}  '
                f'Interval BPM={interval_bpm:7.2f}  '
                f'Beats={result["beat_count"]:3}'
            )

        except Exception as exc:
            print(f"{path.name}: ERROR: {exc}")


if __name__ == "__main__":
    main()
