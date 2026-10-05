from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature2.config import MODEL_DIR, PROCESSED_DIR
from feature2.model import LightingPredictionModel
from feature2.preprocessing import FeatureScaler


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = PROCESSED_DIR / "dataset.npz"
MODEL_PATH = MODEL_DIR / "best_model.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.npz"
METADATA_PATH = MODEL_DIR / "training_metadata.json"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_NAMES = [
    "valence_mean",
    "arousal_mean",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset() -> dict:
    """Load the processed Feature 2 dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature 2 dataset not found:\n{DATASET_PATH}"
        )

    data = np.load(DATASET_PATH, allow_pickle=True)

    required_fields = {
        "song_ids",
        "features",
        "targets",
        "feature_names",
    }

    missing = required_fields - set(data.files)

    if missing:
        raise ValueError(
            f"Dataset is missing required fields: {sorted(missing)}"
        )

    return {
        "song_ids": data["song_ids"],
        "features": data["features"].astype(np.float32),
        "targets": data["targets"].astype(np.float32),
        "feature_names": data["feature_names"],
    }


# ---------------------------------------------------------------------------
# Scaler loading
# ---------------------------------------------------------------------------

def load_scaler() -> FeatureScaler:
    """Load the scaler fitted during model training."""

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Feature scaler not found:\n{SCALER_PATH}"
        )

    scaler_data = np.load(SCALER_PATH)

    scaler = FeatureScaler()

    # Reconstruct sklearn StandardScaler.
    sklearn_scaler = scaler.scaler

    sklearn_scaler.mean_ = scaler_data["mean"]
    sklearn_scaler.scale_ = scaler_data["scale"]

    if "var" in scaler_data:
        sklearn_scaler.var_ = scaler_data["var"]
    else:
        sklearn_scaler.var_ = sklearn_scaler.scale_ ** 2

    sklearn_scaler.n_features_in_ = len(sklearn_scaler.mean_)

    return scaler


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(input_dim: int) -> LightingPredictionModel:
    """Load the best trained Feature 2 model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # The training script stores model configuration in the checkpoint.
    hidden_dim = checkpoint.get("hidden_dim", 256)
    dropout = checkpoint.get("dropout", 0.20)

    model = LightingPredictionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        # Support a checkpoint containing the state dict directly.
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_features(
    features: np.ndarray,
    feature_names: np.ndarray | None = None,
) -> dict:
    """
    Predict valence and arousal from a single 531-dimensional
    Feature 1 feature vector.
    """

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D feature array, got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Input features contain NaN or infinite values.")

    # Load scaler.
    scaler = load_scaler()

    expected_dim = len(scaler.scaler.mean_)

    if features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch.\n"
            f"Expected: {expected_dim}\n"
            f"Received: {features.shape[1]}"
        )

    # Apply exactly the same scaling used during training.
    scaled_features = scaler.transform(features)

    # Load trained model.
    model = load_model(expected_dim)

    tensor = torch.from_numpy(scaled_features).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)

    prediction = output.cpu().numpy()[0]

    return {
        "valence_mean": float(prediction[0]),
        "arousal_mean": float(prediction[1]),
    }


# ---------------------------------------------------------------------------
# Predict a song from the processed dataset
# ---------------------------------------------------------------------------

def predict_song(song_id: str) -> dict:
    """Find a song in dataset.npz and predict its emotional values."""

    data = load_dataset()

    song_ids = np.asarray(data["song_ids"]).astype(str)

    matches = np.where(song_ids == str(song_id))[0]

    if len(matches) == 0:
        raise ValueError(
            f"Song ID '{song_id}' was not found in the Feature 2 dataset."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Song ID '{song_id}' occurs multiple times in the dataset."
        )

    index = int(matches[0])

    features = data["features"][index]

    prediction = predict_features(
        features,
        data["feature_names"],
    )

    prediction["song_id"] = str(song_id)

    return prediction


# ---------------------------------------------------------------------------
# Print prediction
# ---------------------------------------------------------------------------

def print_prediction(result: dict) -> None:
    """Display prediction in a clean format."""

    print()
    print("=" * 60)
    print("FEATURE 2 - LIGHTING PREDICTION")
    print("=" * 60)

    print(f"Song ID       : {result['song_id']}")
    print(f"Valence Mean  : {result['valence_mean']:.6f}")
    print(f"Arousal Mean  : {result['arousal_mean']:.6f}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict valence and arousal using the trained Feature 2 model."
    )

    parser.add_argument(
        "--song-id",
        required=True,
        help="DEAM song ID, for example 2",
    )

    args = parser.parse_args()

    result = predict_song(args.song_id)

    print_prediction(result)


if __name__ == "__main__":
    main()