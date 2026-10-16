
from __future__ import annotations

from typing import Any, Literal

from feature2.lighting_mapper import (
    create_lighting_state,
    lighting_state_to_dict,
)

Mode = Literal["performer", "audience"]


def predict_lighting(
    mode: Mode,
    emotion: dict[str, Any],
    beat: dict[str, Any],
    *,
    gesture: dict[str, Any] | None = None,
    audience: dict[str, Any] | None = None,
    beat_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a stage-lighting prediction from audio and video signals.

    Audio:
        emotion: Feature 2 prediction containing valence_mean and
                 arousal_mean.
        beat:    Offline beat analysis or current live beat metadata.
        beat_event: Optional event from LiveBeatTracker.update().

    Performer mode:
        Uses Feature 4 gesture events.

    Audience mode:
        Uses Feature 3 audience activity information.

    Mood determines the target colour.
    A live beat event determines when to apply the target.

    This function predicts lighting commands. It does not control
    physical lights or run any model itself.
    """

    # 1. Validate mode.
    if mode not in ("performer", "audience"):
        raise ValueError(
            "mode must be 'performer' or 'audience'"
        )

    # 2. Validate and read emotion predictions.
    try:
        valence = float(emotion["valence_mean"])
        arousal = float(emotion["arousal_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "emotion must contain numeric valence_mean "
            "and arousal_mean values"
        ) from exc

    if not (-1.0 <= valence <= 1.0):
        raise ValueError("valence_mean must be between -1 and 1")

    if not (-1.0 <= arousal <= 1.0):
        raise ValueError("arousal_mean must be between -1 and 1")

    # 3. Convert mood into the target lighting colour.
    lighting_state = create_lighting_state(
        valence=valence,
        arousal=arousal,
    )
    lighting = lighting_state_to_dict(lighting_state)

    # 4. Read live beat information.
    beat_event = beat_event or {}

    live_beat = bool(beat_event.get("beat_detected", False))

    beat_timestamp = (
        beat_event.get("timestamp") if live_beat else None
    )

    live_bpm = beat_event.get("current_bpm")

    # Prefer the current live BPM when available.
    bpm = (
        live_bpm
        if live_bpm is not None
        else beat.get("estimated_bpm")
    )

    beat_interval = beat_event.get("beat_interval_seconds")

    if beat_interval is None:
        beat_interval = beat.get(
            "estimated_beat_interval_seconds"
        )

    # 5. Prepare the lighting action.
    # The target colour can be calculated on every prediction.
    # The controller should apply it only when apply_now is True.
    lighting_action: dict[str, Any] = {
        "target_rgb": lighting["rgb"],
        "hue": lighting["hue"],
        "brightness": lighting["brightness"],
        "transition_speed": lighting["transition_speed"],
        "trigger": "beat" if live_beat else "wait",
        "apply_now": live_beat,
        "effect": "normal",
    }

    # 6. Build the shared prediction payload.
    result: dict[str, Any] = {
        "mode": mode,
        "emotion": {
            "valence": valence,
            "arousal": arousal,
        },
        "beat": {
            "bpm": bpm,
            "beat_times_seconds": beat.get(
                "beat_times_seconds", []
            ),
            "beat_interval_seconds": beat_interval,
            "live_beat_detected": live_beat,
            "live_beat_timestamp": beat_timestamp,
            "onset_strength": beat_event.get(
                "onset_strength"
            ),
        },
        "lighting_prediction": lighting,
        "lighting_action": lighting_action,
    }

    # 7. Add context from the selected visual mode only.
    if mode == "performer":
        gesture_event = (
            gesture.get("event") if gesture else None
        )

        direction = (
            gesture.get("direction") if gesture else None
        )

        result["performer"] = {
            "gesture": gesture_event,
            "direction": direction,
        }

        # These are effect requests for the controller.
        # They do not change the mood colour by themselves.
        if gesture_event == "FULL_STAGE_REVEAL":
            lighting_action["effect"] = "full_stage_reveal"

        elif gesture_event == "CLIMAX":
            lighting_action["effect"] = "climax"

        elif gesture_event == "PAUSE_AUTO":
            lighting_action["effect"] = "pause_auto"

        elif gesture_event == "SPOTLIGHT_DIRECTION":
            lighting_action["effect"] = "spotlight"
            lighting_action["direction"] = direction

    else:
        result["audience"] = {
            "energy": (
                audience.get("audience_energy")
                if audience else None
            ),
            "activity_level": (
                audience.get("activity_level")
                if audience else None
            ),
            "person_count": (
                audience.get("person_count")
                if audience else None
            ),
        }

    return result


if __name__ == "__main__":
    # Standalone smoke test using illustrative predictions.
    # No microphone, webcam, or physical lights are required.

    example_emotion = {
        "valence_mean": 0.65,
        "arousal_mean": 0.80,
    }

    example_beat = {
        "estimated_bpm": 120.0,
        "beat_times_seconds": [],
        "estimated_beat_interval_seconds": 0.5,
    }

    example_live_beat = {
        "beat_detected": True,
        "timestamp": 12.48,
        "current_bpm": 120.0,
        "beat_interval_seconds": 0.5,
        "onset_strength": 0.91,
    }

    prediction = predict_lighting(
        mode="performer",
        emotion=example_emotion,
        beat=example_beat,
        beat_event=example_live_beat,
        gesture={"event": "CLIMAX"},
    )

    import json

    print(json.dumps(prediction, indent=2))
