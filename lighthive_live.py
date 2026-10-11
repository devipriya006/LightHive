
import sys
import socket
import json
from pathlib import Path

import numpy as np
import torch


from feature1.audio_features import (
    extract_audio_features,
    load_clap_model,
)

from feature2.predict import load_model, load_scaler
from feature2.lighting_mapper import create_lighting_state


# ============================================================
# SETTINGS
# ============================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Temporary numeric filename because Feature 1 currently
# expects the MP3 filename stem to be numeric.
TEMP_AUDIO_DIR = Path("live_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_live_features(audio_path: Path) -> np.ndarray:

    print("\n[1/5] Extracting classical audio features...")

    # Feature 1 currently expects a numeric filename.
    temp_path = TEMP_AUDIO_DIR / "999999.mp3"

    if temp_path.resolve() != audio_path.resolve():
         if temp_path.exists():
             temp_path.unlink()

    
    temp_path.write_bytes(audio_path.read_bytes())
    classical = extract_audio_features(temp_path)

    # Remove song_id.
    classical.pop("song_id", None)

    features = {}

    # Same flattening logic used by feature2/dataset.py
    for key, value in classical.items():

        if isinstance(value, (int, float)):
            features[key] = float(value)

        elif isinstance(value, list):

            numeric = np.asarray(
                value,
                dtype=np.float32
            )

            if numeric.size == 0:
                continue

            features[f"{key}_mean"] = float(
                numeric.mean()
            )

            features[f"{key}_std"] = float(
                numeric.std()
            )

    print(
        f"Classical features: {len(features)}"
    )

    print("\n[2/5] Loading CLAP model...")

    clap_model = load_clap_model()

    print("\n[3/5] Extracting CLAP embedding...")

    embedding = clap_model.get_audio_embedding_from_filelist(
        x=[str(audio_path)],
        use_tensor=False
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    print(
        f"CLAP embedding: {embedding.shape}"
    )

    # Add CLAP features exactly like dataset.py
    for index, value in enumerate(embedding):
        features[f"clap_{index:03d}"] = float(value)

    vector = np.asarray(
        list(features.values()),
        dtype=np.float32
    )

    print(
        f"\nFinal feature vector: {vector.shape}"
    )

    return vector


# ============================================================
# PREDICT LIGHTING
# ============================================================

def predict_lighting(features):

    print("\n[4/5] Running Feature 2 AI...")

    scaler = load_scaler()

    expected_dim = len(
        scaler.scaler.mean_
    )

    print(
        f"Model expects: {expected_dim} features"
    )

    print(
        f"We produced:  {features.shape[0]} features"
    )

    if features.shape[0] != expected_dim:

        raise ValueError(
            "\nFEATURE SIZE MISMATCH!\n"
            f"Model expects {expected_dim}, "
            f"but live extraction produced "
            f"{features.shape[0]}."
        )

    scaled = scaler.transform(
        features.reshape(1, -1)
    )

    model = load_model(expected_dim)

    tensor = torch.from_numpy(
        scaled
    )

    with torch.inference_mode():

        output = model(tensor)

    prediction = output.numpy()[0]

    valence = float(prediction[0])
    arousal = float(prediction[1])

    print("\nAI PREDICTION")
    print("------------------------------")
    print(f"Valence : {valence:.4f}")
    print(f"Arousal : {arousal:.4f}")

    # Convert valence/arousal into lighting.
    state = create_lighting_state(
        valence=valence,
        arousal=arousal
    )

    print("\nCLR LIGHTING")
    print("------------------------------")
    print(f"Hue        : {state.hue:.2f}")
    print(f"Saturation : {state.saturation:.2f}")
    print(f"Brightness : {state.brightness:.2f}")
    print(
        f"Transition : {state.transition_speed:.2f}"
    )
    print(f"RGB        : {state.rgb}")

    return state


# ============================================================
# SEND TO UNITY
# ============================================================

def send_to_unity(state):

    print("\n[5/5] Sending lighting to Unity...")

    r, g, b = state.rgb

    hex_color = (
        f"#{r:02X}{g:02X}{b:02X}"
    )

    # Convert normalized brightness 0-1
    # to Unity's 0-40 lighting range.
    unity_brightness = (
        state.brightness * 40.0
    )

    command = {
        "color": hex_color,
        "brightness": unity_brightness,
        "transitionSpeed": state.transition_speed,
        "beat": False
    }

    print("\nUNITY COMMAND")
    print("------------------------------")
    print(json.dumps(command, indent=2))

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.sendto(
        json.dumps(command).encode("utf-8"),
        (UDP_IP, UDP_PORT)
    )

    sock.close()

    print("\n✅ Lighting command sent to Unity!")


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python lighthive_live.py "
            "path/to/song.mp3"
        )

        return

    audio_path = Path(
        sys.argv[1]
    ).resolve()

    if not audio_path.exists():

        print(
            f"\n❌ File not found:"
            f"\n{audio_path}"
        )

        return

    print("=" * 60)
    print("LIGHTHIVE LIVE LIGHTING")
    print("=" * 60)

    print(
        f"\nSong: {audio_path.name}"
    )

    try:

        features = build_live_features(
            audio_path
        )

        state = predict_lighting(
            features
        )

        send_to_unity(state)

    except Exception as error:

        print("\n❌ ERROR")
        print("=" * 60)
        print(error)
        print("=" * 60)

        raise


if __name__ == "__main__":
    main()