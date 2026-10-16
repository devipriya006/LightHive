from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AROUSAL_FILENAME,
    ANNOTATION_DIR,
    CLASSICAL_FEATURE_DIR,
    CLAP_DIMENSION,
    CLAP_FEATURE_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SCHEMA_VERSION,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    VALENCE_FILENAME,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetBundle:
    """Validated song-level dataset used by Feature 2."""

    song_ids: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]


def _find_annotation_file(filename: str) -> Path:
    matches = list(ANNOTATION_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {ANNOTATION_DIR}"
        )

    if len(matches) > 1:
        LOGGER.warning(
            "Multiple %s files found. Using: %s",
            filename,
            matches[0],
        )

    return matches[0]


def _load_annotations(filename: str) -> pd.DataFrame:
    path = _find_annotation_file(filename)

    df = pd.read_csv(path)

    if "song_id" not in df.columns:
        raise ValueError(
            f"{path} does not contain required 'song_id' column."
        )

    df["song_id"] = df["song_id"].astype(str)

    return df


def _aggregate_annotations(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Convert DEAM's wide temporal annotation format into
    song-level statistical targets.

    Each song receives:
        <prefix>_mean
        <prefix>_std
    """

    sample_columns = [
        column
        for column in df.columns
        if column.startswith("sample_")
    ]

    if not sample_columns:
        raise ValueError(
            f"No temporal sample columns found for {prefix} annotations."
        )

    values = df[sample_columns].apply(pd.to_numeric, errors="coerce")

    result = pd.DataFrame(
        {
            "song_id": df["song_id"].astype(str),
            f"{prefix}_mean": values.mean(axis=1),
            f"{prefix}_std": values.std(axis=1).fillna(0.0),
        }
    )

    return result


def _load_classical_features(song_id: str) -> dict[str, float]:
    path = CLASSICAL_FEATURE_DIR / f"{song_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing classical feature file: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Classical feature file is not a JSON object: {path}"
        )

    return data


def _flatten_classical_features(
    data: dict[str, object],
) -> dict[str, float]:
    """
    Convert the Feature 1 classical JSON representation into
    scalar model inputs.

    Vector features such as MFCC are represented using their
    aggregate statistics rather than expanding them blindly.
    """

    output: dict[str, float] = {}

    for key, value in data.items():
        if key == "song_id":
            continue

        if isinstance(value, (int, float)):
            output[key] = float(value)
            continue

        if isinstance(value, list):
            numeric = np.asarray(value, dtype=np.float32)

            if numeric.size == 0:
                continue

            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"Non-finite values found in classical feature '{key}'."
                )

            output[f"{key}_mean"] = float(numeric.mean())
            output[f"{key}_std"] = float(numeric.std())

    return output


def _load_clap_embedding(song_id: str) -> np.ndarray:
    path = CLAP_FEATURE_DIR / f"{song_id}.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing CLAP embedding: {path}"
        )

    embedding = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.float32,
    ).reshape(-1)

    if embedding.shape[0] != CLAP_DIMENSION:
        raise ValueError(
            f"{path} has dimension {embedding.shape[0]}, "
            f"expected {CLAP_DIMENSION}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            f"Non-finite values found in {path}."
        )

    return embedding


def build_dataset() -> DatasetBundle:
    """
    Build a validated song-level Feature 2 dataset.

    Returns:
        DatasetBundle containing:
            - song IDs
            - model input matrix
            - target matrix
            - feature names
    """

    arousal = _load_annotations(AROUSAL_FILENAME)
    valence = _load_annotations(VALENCE_FILENAME)

    arousal_targets = _aggregate_annotations(arousal, "arousal")
    valence_targets = _aggregate_annotations(valence, "valence")

    annotations = arousal_targets.merge(
        valence_targets,
        on="song_id",
        how="inner",
        validate="one_to_one",
    )

    records: list[dict[str, float | str]] = []

    for song_id in annotations["song_id"]:
        classical = _flatten_classical_features(
            _load_classical_features(song_id)
        )

        clap = _load_clap_embedding(song_id)

        record: dict[str, float | str] = {
            "song_id": song_id,
            **classical,
        }

        for index, value in enumerate(clap):
            record[f"clap_{index:03d}"] = float(value)

        records.append(record)

    feature_df = pd.DataFrame(records)

    if feature_df.empty:
        raise RuntimeError("Feature 2 dataset is empty.")

    merged = annotations.merge(
        feature_df,
        on="song_id",
        how="inner",
        validate="one_to_one",
    )

    feature_names = [
        column
        for column in feature_df.columns
        if column != "song_id"
    ]

    target_names = [
        "valence_mean",
        "valence_std",
        "arousal_mean",
        "arousal_std",
    ]

    X = merged[feature_names].to_numpy(dtype=np.float32)
    y = merged[target_names].to_numpy(dtype=np.float32)
    song_ids = merged["song_id"].to_numpy(dtype=str)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    if not np.isfinite(y).all():
        raise ValueError("Target matrix contains NaN or infinite values.")

    if len(song_ids) != len(np.unique(song_ids)):
        raise ValueError("Duplicate song IDs detected.")

    LOGGER.info(
        "Feature 2 dataset built: samples=%d, features=%d, targets=%d",
        X.shape[0],
        X.shape[1],
        y.shape[1],
    )

    return DatasetBundle(
        song_ids=song_ids,
        features=X,
        targets=y,
        feature_names=tuple(feature_names),
    )


def save_dataset(bundle: DatasetBundle) -> Path:
    """Persist the validated dataset as a compressed NumPy artifact."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "dataset.npz"

    np.savez_compressed(
        output_path,
        song_ids=bundle.song_ids,
        features=bundle.features,
        targets=bundle.targets,
        feature_names=np.asarray(bundle.feature_names),
        schema_version=np.asarray(SCHEMA_VERSION),
    )

    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    bundle = build_dataset()
    path = save_dataset(bundle)

    print("\nFeature 2 dataset created successfully.")
    print(f"Samples : {bundle.features.shape[0]}")
    print(f"Features: {bundle.features.shape[1]}")
    print(f"Targets : {bundle.targets.shape[1]}")
    print(f"Saved   : {path}")
    