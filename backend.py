import os
import tempfile
import librosa

from pathlib import Path
from fastapi import UploadFile, File, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import numpy as np
from feature1.audio_features import extract_audio_features, load_clap_model
from feature2.predict import predict_features
from feature2.lighting_mapper import (
    create_lighting_state,
    lighting_state_to_dict,
)

app = FastAPI(title="LightHive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
CLAP_MODEL = None

@app.on_event("startup")
def startup_event():
    global CLAP_MODEL
    CLAP_MODEL = load_clap_model()

class LightingInput(BaseModel):
    valence: float = 0.0
    arousal: float = 0.0
    audience_energy: float = 0.0
    beat: bool = False

@app.get("/")
def home():
    return {"message": "LightHive API is running"}

@app.get("/state")
def get_state():
    state = lighting_state_to_dict(
        create_lighting_state(0.0, 0.0)
    )
    state.update({
        "valence": 0.0,
        "arousal": 0.0,
        "audience_energy": 0.0,
        "beat": False,
        "mode": "preview",
    })
    return state

@app.post("/state")
def update_state(data: LightingInput):
    state = lighting_state_to_dict(
        create_lighting_state(data.valence, data.arousal)
    )
    state.update({
        "valence": data.valence,
        "arousal": data.arousal,
        "audience_energy": data.audience_energy,
        "beat": data.beat,
        "mode": "preview",
    })
    return state


def build_model_features(audio_path):
    """Build model inputs in the exact order recorded in training metadata."""
    metadata_path = Path(__file__).resolve().parent / "models" / "feature2" / "training_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    classical = extract_audio_features(Path(audio_path))
    classical.pop("song_id", None)

    values = {}

    # Match feature2/dataset.py's classical feature flattening.
    for key, value in classical.items():
        if isinstance(value, list):
            numeric = np.asarray(value, dtype=np.float32)
            values[f"{key}_mean"] = float(numeric.mean())
            values[f"{key}_std"] = float(numeric.std())
        else:
            values[key] = float(value)

    # Generate the 512-dimensional CLAP embedding.
    embedding = CLAP_MODEL.get_audio_embedding_from_filelist(
        x=[str(audio_path)],
        use_tensor=False,
    )
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if embedding.size != 512:
        raise ValueError(
            f"Expected 512 CLAP values, received {embedding.size}"
        )

    for index, value in enumerate(embedding):
        values[f"clap_{index:03d}"] = float(value)

    feature_names = metadata["feature_names"]
    missing = [name for name in feature_names if name not in values]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    vector = np.asarray(
        [values[name] for name in feature_names],
        dtype=np.float32,
    )

    if vector.shape != (531,):
        raise ValueError(f"Expected 531 features, got {vector.shape}")

    return vector

@app.post("/predict-lighting")
async def predict_lighting(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    extension = Path(file.filename).suffix.lower()
    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
            temp_path = temp.name
            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        # Extract features and predict emotions.
        features = build_model_features(temp_path)
        prediction = predict_features(features)

        valence = prediction["valence_mean"]
        arousal = prediction["arousal_mean"]

        # Convert predicted emotions into lighting values.
        state = lighting_state_to_dict(
            create_lighting_state(valence, arousal)
        )

        return {
            "filename": file.filename,
            "valence": valence,
            "arousal": arousal,
            "mode": "ml_prediction",
            "lighting": state,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Lighting prediction failed: {str(exc)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False
        ) as temp:
            temp_path = temp.name
            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        y, sr = librosa.load(temp_path, sr=22050, mono=True)

        if len(y) == 0:
            raise HTTPException(
                status_code=400, detail="Audio file is empty"
            )

        duration = len(y) / sr
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr
        )
        rms = librosa.feature.rms(y=y)[0]

        return {
            "filename": file.filename,
            "duration_seconds": round(float(duration), 2),
            "tempo_bpm": round(float(tempo.item()), 2),
            "beat_count": int(len(beat_frames)),
            "rms_mean": round(float(rms.mean()), 4),
            "mode": "analysis_only",
            "message": "Audio features extracted successfully"
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Audio analysis failed: {str(exc)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()

