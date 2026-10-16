import numpy as np
import torch

from pathlib import Path
from feature2.model import LightingPredictionModel
def predict_features(features) -> dict[str, float]:
    """
    Predict song-level valence and arousal using the trained model.

    This function expects the existing trained model files and
    the same 531-feature ordering used during training.
    """
    import joblib
    import numpy as np
    from pathlib import Path

    model_dir = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "feature2"
    )

    # Update these filenames if your saved model uses different names.
    valence_path = model_dir / "valence_model.pkl"
    arousal_path = model_dir / "arousal_model.pkl"

    if not valence_path.exists() or not arousal_path.exists():
        raise FileNotFoundError(
            "Could not find valence_model.pkl and arousal_model.pkl "
            f"in {model_dir}. Check your actual model filenames."
        )

    feature_vector = np.asarray(features, dtype=np.float32).reshape(1, -1)

    if feature_vector.shape[1] != 531:
        raise ValueError(
            f"Expected 531 features, got {feature_vector.shape[1]}"
        )

    valence_model = joblib.load(valence_path)
    arousal_model = joblib.load(arousal_path)

    valence = float(np.asarray(valence_model.predict(feature_vector)).ravel()[0])
    arousal = float(np.asarray(arousal_model.predict(feature_vector)).ravel()[0])

    return {
        "valence_mean": max(-1.0, min(1.0, valence)),
        "arousal_mean": max(-1.0, min(1.0, arousal)),
    }
    
def predict_features(features) -> dict[str, float]:
    """Predict valence and arousal using the saved Feature 2 model."""

    model_dir = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "feature2"
    )

    model_path = model_dir / "best_model.pt"
    scaler_path = model_dir / "feature_scaler.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint created by the training script.
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unexpected checkpoint format: expected model_state_dict."
        )

    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    dropout = float(checkpoint["dropout"])

    feature_vector = np.asarray(features, dtype=np.float32).reshape(-1)

    if feature_vector.size != input_dim:
        raise ValueError(
            f"Expected {input_dim} features, got {feature_vector.size}."
        )

    # Apply the exact scaler values saved by the training script.
    with np.load(scaler_path) as scaler_data:
        mean = np.asarray(scaler_data["mean"], dtype=np.float32)
        scale = np.asarray(scaler_data["scale"], dtype=np.float32)

    if mean.size != input_dim or scale.size != input_dim:
        raise ValueError("Scaler dimensions do not match model input.")

    if not np.isfinite(feature_vector).all():
        raise ValueError("Input features contain NaN or infinity.")

    # Protect against zero or invalid scale values.
    scale = np.where(
        np.isfinite(scale) & (scale > 0),
        scale,
        1.0,
    )

    scaled_features = (feature_vector - mean) / scale

    model = LightingPredictionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tensor = torch.from_numpy(
        scaled_features.astype(np.float32)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = output.detach().cpu().numpy().reshape(-1)

    if prediction.size != 2:
        raise ValueError(
            f"Expected 2 model outputs, got {prediction.size}."
        )

    valence, arousal = map(float, prediction)

    # The training targets are valence_mean and arousal_mean.
    # Clamp only if your target scale is defined as [-1, 1].
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    return {
        "valence_mean": valence,
        "arousal_mean": arousal,
    }
