from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LightingState:
    """
    Virtual lighting state produced from Feature 2 predictions.
    """

    hue: float
    saturation: float
    brightness: float
    transition_speed: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Convert HSV lighting representation to RGB."""

        h = self.hue / 360.0
        s = self.saturation
        v = self.brightness

        i = int(h * 6.0)
        f = h * 6.0 - i

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        i %= 6

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        return (
            round(r * 255),
            round(g * 255),
            round(b * 255),
        )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_emotion(value: float) -> float:
    """
    Convert DEAM-style emotion values approximately from [-1, 1]
    into [0, 1].
    """

    return clamp((value + 1.0) / 2.0, 0.0, 1.0)


def map_valence_to_hue(valence: float) -> float:
    """
    Map valence to a continuous hue.

    Lower valence  -> cooler/darker emotional region.
    Higher valence -> warmer/brighter emotional region.
    """

    normalized = normalize_emotion(valence)

    # 240° = blue
    # 0°   = red

    return 240.0 * (1.0 - normalized)


def map_arousal_to_brightness(arousal: float) -> float:
    """
    Map arousal to lighting brightness.

    Low arousal  -> dimmer
    High arousal -> brighter
    """

    normalized = normalize_emotion(arousal)

    minimum_brightness = 0.20
    maximum_brightness = 1.00

    return (
        minimum_brightness
        + normalized
        * (maximum_brightness - minimum_brightness)
    )


def map_arousal_to_transition(arousal: float) -> float:
    """
    Map arousal to transition speed.

    Returns transitions per second.

    Low arousal  -> slow changes
    High arousal -> fast changes
    """

    normalized = normalize_emotion(arousal)

    minimum_speed = 0.25
    maximum_speed = 4.0

    return (
        minimum_speed
        + normalized
        * (maximum_speed - minimum_speed)
    )


def create_lighting_state(
    valence: float,
    arousal: float,
) -> LightingState:
    """
    Convert Feature 2 predictions into a virtual lighting state.
    """

    hue = map_valence_to_hue(valence)

    brightness = map_arousal_to_brightness(arousal)

    transition_speed = map_arousal_to_transition(arousal)

    # Keep saturation high enough for stage lighting while
    # avoiding completely artificial maximum saturation.
    saturation = 0.85

    return LightingState(
        hue=hue,
        saturation=saturation,
        brightness=brightness,
        transition_speed=transition_speed,
    )


def lighting_state_to_dict(
    state: LightingState,
) -> dict:
    """Return JSON/WebSocket-friendly lighting state."""

    rgb = state.rgb

    return {
        "hue": round(state.hue, 2),
        "saturation": round(state.saturation, 3),
        "brightness": round(state.brightness, 3),
        "transition_speed": round(state.transition_speed, 3),
        "rgb": {
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
        },
    }


if __name__ == "__main__":

    # Example using the prediction from Song ID 2.
    valence = -0.224702
    arousal = -0.228699

    state = create_lighting_state(
        valence=valence,
        arousal=arousal,
    )

    print("=" * 60)
    print("FEATURE 2 - LIGHTING STATE")
    print("=" * 60)

    print(f"Valence            : {valence:.6f}")
    print(f"Arousal            : {arousal:.6f}")
    print(f"Hue                : {state.hue:.2f}°")
    print(f"Saturation         : {state.saturation:.3f}")
    print(f"Brightness         : {state.brightness:.3f}")
    print(f"Transition Speed   : {state.transition_speed:.3f}/sec")
    print(f"RGB                : {state.rgb}")

    print("=" * 60)