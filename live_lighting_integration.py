
from __future__ import annotations

import numpy as np

from feature2.live_beat_tracker import LiveBeatTracker
from final_predictor import predict_lighting


SAMPLE_RATE = 22050
CHUNK_SECONDS = 0.25


def run_live_lighting(
    audio_chunks,
    emotion: dict,
    mode: str = "performer",
):
    tracker = LiveBeatTracker(sample_rate=SAMPLE_RATE)

    beat_info = {
        "estimated_bpm": None,
        "estimated_beat_interval_seconds": None,
        "beat_times_seconds": [],
    }

    for chunk in audio_chunks:
        beat_event = tracker.update(np.asarray(chunk, dtype=np.float32))

        if beat_event["current_bpm"] is not None:
            beat_info["estimated_bpm"] = beat_event["current_bpm"]
            beat_info["estimated_beat_interval_seconds"] = (
                beat_event["beat_interval_seconds"]
            )

        result = predict_lighting(
            mode=mode,
            emotion=emotion,
            beat=beat_info,
            beat_event=beat_event,
        )

        if beat_event["beat_detected"]:
            print(
                f"Beat at {beat_event['timestamp']}s | "
                f"BPM: {beat_event['current_bpm']} | "
                f"Lighting: {result['lighting_action']['target_rgb']}"
            )

        yield result


