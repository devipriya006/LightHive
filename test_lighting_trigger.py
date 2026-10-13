from final_predictor import predict_lighting

emotion = {
    "valence_mean": 0.65,
    "arousal_mean": 0.80,
}

beat = {
    "estimated_bpm": 93.96,
    "estimated_beat_interval_seconds": 0.639,
    "beat_times_seconds": [],
}

beat_event = {
    "beat_detected": True,
    "timestamp": 12.48,
    "current_bpm": 93.96,
    "beat_interval_seconds": 0.639,
    "onset_strength": 0.91,
}

result = predict_lighting(
    mode="performer",
    emotion=emotion,
    beat=beat,
    beat_event=beat_event,
    gesture=None,
)

print("Lighting action:", result["lighting_action"])