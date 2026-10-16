
import os
import tempfile
import librosa
import cv2
import json
import numpy as np
import mediapipe as mp

from pathlib import Path
from fastapi import UploadFile, File, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature1.audio_features import extract_audio_features, load_clap_model
from feature2.predict import predict_features
from feature2.beat_detector import detect_beats
from feature2.lighting_mapper import (
    create_lighting_state,
    lighting_state_to_dict,
)
from feature3.audience_energy import AudienceEnergyAnalyzer
from feature4.gesture_detector import GestureDetector


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
GESTURE_DETECTOR = None
POSE_TRACKER = None
AUDIENCE_ANALYZER = None


@app.on_event("startup")
def startup_event():
    global CLAP_MODEL, GESTURE_DETECTOR, POSE_TRACKER, AUDIENCE_ANALYZER

    CLAP_MODEL = load_clap_model()
    GESTURE_DETECTOR = GestureDetector()

    AUDIENCE_ANALYZER = AudienceEnergyAnalyzer(
        model_path="yolo11n-pose.pt",
        image_size=416,
        max_people=10,
        device="cpu",
    )

    POSE_TRACKER = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@app.on_event("shutdown")
def shutdown_event():
    global GESTURE_DETECTOR, POSE_TRACKER

    if GESTURE_DETECTOR is not None:
        GESTURE_DETECTOR.close()
        GESTURE_DETECTOR = None

    if POSE_TRACKER is not None:
        POSE_TRACKER.close()
        POSE_TRACKER = None


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
    """Build model inputs in the feature order used during training."""
    metadata_path = (
        Path(__file__).resolve().parent
        / "models"
        / "feature2"
        / "training_metadata.json"
    )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    classical = extract_audio_features(Path(audio_path))
    classical.pop("song_id", None)

    values = {}

    for key, value in classical.items():
        if isinstance(value, list):
            numeric = np.asarray(value, dtype=np.float32)
            values[f"{key}_mean"] = float(numeric.mean())
            values[f"{key}_std"] = float(numeric.std())
        else:
            values[key] = float(value)

    if CLAP_MODEL is None:
        raise RuntimeError("CLAP model is not initialized")

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
        raise HTTPException(
            status_code=400,
            detail="No file selected",
        )

    extension = Path(file.filename).suffix.lower()
    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp_path = temp.name

            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        features = build_model_features(temp_path)
        prediction = predict_features(features)

        valence = float(prediction["valence_mean"])
        arousal = float(prediction["arousal_mean"])

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


@app.post("/detect-beats")
async def detect_beats_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        return detect_beats(temp_path)

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@app.post("/detect-pose")
async def detect_pose(file: UploadFile = File(...)):
    if POSE_TRACKER is None:
        raise HTTPException(
            status_code=503,
            detail="Pose tracker is not ready",
        )

    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image frame",
        )

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = POSE_TRACKER.process(rgb_frame)

        if not result.pose_landmarks:
            return {
                "detected": False,
                "landmarks": None,
            }

        landmarks = result.pose_landmarks.landmark

        joint_names = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }

        points = {}

        for name, index in joint_names.items():
            point = landmarks[index]
            points[name] = {
                "x": round(float(point.x), 4),
                "y": round(float(point.y), 4),
                "visibility": round(float(point.visibility), 4),
            }

        return {
            "detected": True,
            "landmarks": points,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pose detection failed: {exc}",
        )
    finally:
        await file.close()


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected",
        )

    allowed_extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp:
            temp_path = temp.name

            while chunk := await file.read(1024 * 1024):
                temp.write(chunk)

        y, sr = librosa.load(
            temp_path,
            sr=22050,
            mono=True,
        )

        if len(y) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty",
            )

        duration = len(y) / sr

        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
        )

        hop_length = 512
        frame_length = 2048

        # Frame-by-frame audio energy.
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # Frame-by-frame onset strength.
        onset = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        # Calculate separate timestamps because the feature arrays
        # may not have exactly the same number of frames.
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length,
        )

        onset_times = librosa.frames_to_time(
            np.arange(len(onset)),
            sr=sr,
            hop_length=hop_length,
        )

        # Three-second windows, advancing every second.
        window_seconds = 3.0
        step_seconds = 1.0
        window_features = []

        start = 0.0

        while start < duration:
            end = min(start + window_seconds, duration)

            rms_mask = (
                (rms_times >= start)
                & (rms_times < end)
            )

            onset_mask = (
                (onset_times >= start)
                & (onset_times < end)
            )

            energy = (
                float(np.mean(rms[rms_mask]))
                if np.any(rms_mask)
                else 0.0
            )

            onset_strength = (
                float(np.mean(onset[onset_mask]))
                if np.any(onset_mask)
                else 0.0
            )

            window_features.append({
                "start_seconds": round(float(start), 2),
                "end_seconds": round(float(end), 2),
                "energy": round(energy, 4),
                "onset_strength": round(onset_strength, 4),
            })

            start += step_seconds

        return {
            "filename": file.filename,
            "duration_seconds": round(float(duration), 2),
            "tempo_bpm": round(float(np.asarray(tempo).item()), 2),
            "beat_count": int(len(beat_frames)),
            "rms_mean": round(float(rms.mean()), 4),
            "window_features": window_features,
            "mode": "analysis_only",
            "message": "Audio features extracted successfully",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Audio analysis failed: {str(exc)}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        await file.close()


@app.post("/detect-gesture")
async def detect_gesture(file: UploadFile = File(...)):
    if GESTURE_DETECTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Gesture detector is not ready",
        )

    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image frame",
        )

    try:
        result = GESTURE_DETECTOR.detect(frame)

        return {
            "type": result.get("type"),
            "detected": result.get("event"),
            "pause_active": result.get("pause_active", False),
            "direction": result.get("direction"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gesture detection failed: {exc}",
        )
    finally:
        await file.close()


@app.post("/analyze-audience")
async def analyze_audience(file: UploadFile = File(...)):
    if AUDIENCE_ANALYZER is None:
        raise HTTPException(
            status_code=503,
            detail="Audience analyzer is not ready",
        )

    contents = await file.read()
    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image frame",
        )

    try:
        _, payload = AUDIENCE_ANALYZER.update(frame)
        return payload

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audience analysis failed: {exc}",
        )
    finally:
        await file.close()
