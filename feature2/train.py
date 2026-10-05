from __future__ import annotations

import json
import logging
import random

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature2.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    TEST_RATIO,
    VALIDATION_RATIO,
    WEIGHT_DECAY,
)
from feature2.model import LightingPredictionModel
from feature2.preprocessing import FeatureScaler


# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = PROCESSED_DIR / "dataset.npz"

MODEL_PATH = MODEL_DIR / "best_model.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.npz"
SPLITS_PATH = MODEL_DIR / "splits.json"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"

# dataset.py target order:
#
#   0 = valence_mean
#   1 = valence_std
#   2 = arousal_mean
#   3 = arousal_std
#
# The neural network predicts the two actual supervised targets.
TARGET_INDICES = [0, 2]

TARGET_NAMES = [
    "valence_mean",
    "arousal_mean",
]

HIDDEN_DIM = 256
DROPOUT = 0.20

MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility over maximum GPU performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load and validate the Feature 2 dataset.

    Expected dataset.npz fields:

        song_ids
        features
        targets
        feature_names
        schema_version

    Expected target order:

        [valence_mean,
         valence_std,
         arousal_mean,
         arousal_std]

    The model uses only:

        [valence_mean, arousal_mean]
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n\n"
            "Run feature2/dataset.py first."
        )

    data = np.load(
        DATASET_PATH,
        allow_pickle=False,
    )

    logger.info(
        "Dataset fields: %s",
        data.files,
    )

    # -------------------------------------------------------------------------
    # Validate required fields
    # -------------------------------------------------------------------------

    required_fields = {
        "song_ids",
        "features",
        "targets",
        "feature_names",
    }

    missing_fields = required_fields.difference(data.files)

    if missing_fields:
        raise ValueError(
            "Dataset is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    # -------------------------------------------------------------------------
    # Load arrays
    # -------------------------------------------------------------------------

    X = np.asarray(
        data["features"],
        dtype=np.float32,
    )

    y_all = np.asarray(
        data["targets"],
        dtype=np.float32,
    )

    song_ids = np.asarray(
        data["song_ids"]
    ).astype(str)

    feature_names = np.asarray(
        data["feature_names"]
    ).astype(str)

    # -------------------------------------------------------------------------
    # Validate dimensions
    # -------------------------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"Features must be 2-dimensional. "
            f"Got shape: {X.shape}"
        )

    if y_all.ndim != 2:
        raise ValueError(
            f"Targets must be 2-dimensional. "
            f"Got shape: {y_all.shape}"
        )

    if X.shape[0] != y_all.shape[0]:
        raise ValueError(
            "Feature/target sample mismatch: "
            f"{X.shape[0]} features vs "
            f"{y_all.shape[0]} targets."
        )

    if len(song_ids) != X.shape[0]:
        raise ValueError(
            "song_ids count does not match feature samples."
        )

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names count does not match feature dimension."
        )

    # -------------------------------------------------------------------------
    # Validate target schema
    # -------------------------------------------------------------------------

    if y_all.shape[1] < 4:
        raise ValueError(
            "Expected at least 4 target columns in the dataset:\n"
            "[valence_mean, valence_std, arousal_mean, arousal_std]\n"
            f"Found {y_all.shape[1]}."
        )

    # Select the actual prediction targets.
    y = y_all[:, TARGET_INDICES]

    # -------------------------------------------------------------------------
    # Numerical validation
    # -------------------------------------------------------------------------

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "Target matrix contains NaN or infinite values."
        )

    # -------------------------------------------------------------------------
    # Song ID validation
    # -------------------------------------------------------------------------

    unique_song_ids = np.unique(song_ids)

    if len(unique_song_ids) != len(song_ids):
        raise ValueError(
            "Duplicate song IDs detected in dataset."
        )

    logger.info(
        "Dataset loaded successfully."
    )

    logger.info(
        "Samples  : %d",
        X.shape[0],
    )

    logger.info(
        "Features : %d",
        X.shape[1],
    )

    logger.info(
        "Targets  : %d",
        y.shape[1],
    )

    logger.info(
        "Targets used: %s",
        TARGET_NAMES,
    )

    return (
        X,
        y,
        song_ids,
        feature_names,
    )


# =============================================================================
# Dataset Splitting
# =============================================================================

def create_splits(
    song_ids: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create deterministic train/validation/test splits.

    Since each song appears exactly once in the dataset, splitting by
    song index prevents accidental overlap between partitions.
    """

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(
        total_ratio,
        1.0,
    ):
        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO "
            "must equal 1.0."
        )

    indices = np.arange(
        len(song_ids)
    )

    # -------------------------------------------------------------------------
    # First split: training vs temporary
    # -------------------------------------------------------------------------

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # -------------------------------------------------------------------------
    # Second split: validation vs test
    # -------------------------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    validation_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # Keep ordering deterministic.
    train_indices = np.sort(
        train_indices
    )

    validation_indices = np.sort(
        validation_indices
    )

    test_indices = np.sort(
        test_indices
    )

    # -------------------------------------------------------------------------
    # Leakage validation
    # -------------------------------------------------------------------------

    train_set = set(train_indices)
    validation_set = set(validation_indices)
    test_set = set(test_indices)

    if train_set & validation_set:
        raise RuntimeError(
            "Data leakage detected between train and validation."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Data leakage detected between train and test."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Data leakage detected between validation and test."
        )

    logger.info(
        "Split sizes:"
    )

    logger.info(
        "  Train      : %d",
        len(train_indices),
    )

    logger.info(
        "  Validation : %d",
        len(validation_indices),
    )

    logger.info(
        "  Test       : %d",
        len(test_indices),
    )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE and R² for each target.
    """

    results: dict[str, float] = {}

    for index, target_name in enumerate(
        TARGET_NAMES
    ):

        y_true = targets[:, index]
        y_pred = predictions[:, index]

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        results[
            f"{target_name}_mae"
        ] = float(mae)

        results[
            f"{target_name}_rmse"
        ] = float(rmse)

        results[
            f"{target_name}_r2"
        ] = float(r2)

    # Overall metrics across both outputs.
    results["overall_mae"] = float(
        mean_absolute_error(
            targets,
            predictions,
        )
    )

    results["overall_rmse"] = float(
        np.sqrt(
            mean_squared_error(
                targets,
                predictions,
            )
        )
    )

    return results


# =============================================================================
# Epoch Execution
# =============================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    Run one training or evaluation epoch.

    If optimizer is provided:
        training mode

    If optimizer is None:
        evaluation mode
    """

    is_training = (
        optimizer is not None
    )

    model.train(
        is_training
    )

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    for features, targets in loader:

        features = features.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):

            predictions = model(
                features
            )

            loss = criterion(
                predictions,
                targets,
            )

            if is_training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()

        batch_size = features.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        all_predictions.append(
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        all_targets.append(
            targets
            .detach()
            .cpu()
            .numpy()
        )

    if total_samples == 0:
        raise RuntimeError(
            "DataLoader produced zero samples."
        )

    predictions_array = np.concatenate(
        all_predictions,
        axis=0,
    )

    targets_array = np.concatenate(
        all_targets,
        axis=0,
    )

    average_loss = (
        total_loss
        / total_samples
    )

    return (
        average_loss,
        predictions_array,
        targets_array,
    )


# =============================================================================
# Main Training
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "FEATURE 2 - LIGHTING PREDICTION MODEL TRAINING"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "Device: %s",
        DEVICE,
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    (
        X,
        y,
        song_ids,
        feature_names,
    ) = load_dataset()

    # -------------------------------------------------------------------------
    # Create splits
    # -------------------------------------------------------------------------

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = create_splits(
        song_ids
    )

    X_train = X[
        train_indices
    ]

    X_validation = X[
        validation_indices
    ]

    X_test = X[
        test_indices
    ]

    y_train = y[
        train_indices
    ]

    y_validation = y[
        validation_indices
    ]

    y_test = y[
        test_indices
    ]

    # -------------------------------------------------------------------------
    # Feature scaling
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on the training set.
    # -------------------------------------------------------------------------

    scaler = FeatureScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -------------------------------------------------------------------------
    # Save scaler
    # -------------------------------------------------------------------------

    np.savez_compressed(
        SCALER_PATH,
        mean=scaler.scaler.mean_.astype(
            np.float32
        ),
        scale=scaler.scaler.scale_.astype(
            np.float32
        ),
        var=scaler.scaler.var_.astype(
            np.float32
        ),
        n_features_in=np.array(
            [X.shape[1]],
            dtype=np.int64,
        ),
    )

    logger.info(
        "Feature scaler saved: %s",
        SCALER_PATH,
    )

    # -------------------------------------------------------------------------
    # PyTorch datasets
    # -------------------------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train_scaled
        ),
        torch.from_numpy(
            y_train
        ),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            X_validation_scaled
        ),
        torch.from_numpy(
            y_validation
        ),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(
            X_test_scaled
        ),
        torch.from_numpy(
            y_test
        ),
    )

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    model = LightingPredictionModel(
        input_dim=X.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info(
        "Model input dimension : %d",
        X.shape[1],
    )

    logger.info(
        "Model output dimension: %d",
        len(TARGET_NAMES),
    )

    logger.info(
        "Trainable parameters   : %d",
        trainable_parameters,
    )

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    criterion = nn.SmoothL1Loss()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------------------
    # Learning-rate scheduler
    # -------------------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    logger.info(
        "=" * 72
    )

    logger.info(
        "Starting training..."
    )

    logger.info(
        "=" * 72
    )

    # =========================================================================
    # Training loop
    # =========================================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ---------------------------------------------------------------------
        # Training epoch
        # ---------------------------------------------------------------------

        train_loss, train_predictions, train_targets = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        # ---------------------------------------------------------------------
        # Validation epoch
        # ---------------------------------------------------------------------

        validation_loss, validation_predictions, validation_targets = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=None,
        )

        # ---------------------------------------------------------------------
        # Scheduler
        # ---------------------------------------------------------------------

        scheduler.step(
            validation_loss
        )

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        train_metrics = calculate_metrics(
            train_targets,
            train_predictions,
        )

        validation_metrics = calculate_metrics(
            validation_targets,
            validation_predictions,
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        # ---------------------------------------------------------------------
        # Save history record
        # ---------------------------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }

        history.append(
            epoch_record
        )

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        logger.info(
            "Epoch %03d/%03d | "
            "train_loss=%.6f | "
            "val_loss=%.6f | "
            "val_mae=%.6f | "
            "valence_R2=%.4f | "
            "arousal_R2=%.4f | "
            "lr=%.2e",
            epoch,
            MAX_EPOCHS,
            train_loss,
            validation_loss,
            validation_metrics[
                "overall_mae"
            ],
            validation_metrics[
                "valence_mean_r2"
            ],
            validation_metrics[
                "arousal_mean_r2"
            ],
            current_learning_rate,
        )

        # ---------------------------------------------------------------------
        # Best model checkpoint
        # ---------------------------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),

                "input_dim": X.shape[1],

                "hidden_dim": HIDDEN_DIM,

                "dropout": DROPOUT,

                "target_names": TARGET_NAMES,

                "target_indices": TARGET_INDICES,

                "feature_names": feature_names.tolist(),

                "schema_version": "feature2-v1",

                "random_seed": RANDOM_SEED,

                "best_validation_loss": float(
                    best_validation_loss
                ),

                "epoch": epoch,
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            logger.info(
                "  -> Best model saved "
                "(epoch %d, val_loss=%.6f)",
                epoch,
                validation_loss,
            )

        else:

            epochs_without_improvement += 1

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            logger.info(
                "Early stopping triggered at epoch %d.",
                epoch,
            )

            logger.info(
                "No validation improvement for %d epochs.",
                EARLY_STOPPING_PATIENCE,
            )

            break

    # =========================================================================
    # Save training history
    # =========================================================================

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    logger.info(
        "Training history saved: %s",
        HISTORY_PATH,
    )

    # =========================================================================
    # Load best checkpoint
    # =========================================================================

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Best model checkpoint was not created."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =========================================================================
    # Final test evaluation
    # =========================================================================

    (
        test_loss,
        test_predictions,
        test_targets,
    ) = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_predictions,
    )

    # =========================================================================
    # Final results
    # =========================================================================

    logger.info(
        ""
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "FINAL TEST RESULTS"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "Test loss: %.6f",
        test_loss,
    )

    for target_name in TARGET_NAMES:

        logger.info(
            "%s | MAE=%.6f | RMSE=%.6f | R2=%.6f",
            target_name,
            test_metrics[
                f"{target_name}_mae"
            ],
            test_metrics[
                f"{target_name}_rmse"
            ],
            test_metrics[
                f"{target_name}_r2"
            ],
        )

    logger.info(
        "Overall | MAE=%.6f | RMSE=%.6f",
        test_metrics[
            "overall_mae"
        ],
        test_metrics[
            "overall_rmse"
        ],
    )

    # =========================================================================
    # Save exact split information
    # =========================================================================

    split_data = {
        "random_seed": RANDOM_SEED,

        "train_song_ids": song_ids[
            train_indices
        ].tolist(),

        "validation_song_ids": song_ids[
            validation_indices
        ].tolist(),

        "test_song_ids": song_ids[
            test_indices
        ].tolist(),
    }

    with SPLITS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split_data,
            file,
            indent=2,
        )

    logger.info(
        "Dataset splits saved: %s",
        SPLITS_PATH,
    )

    # =========================================================================
    # Save training metadata
    # =========================================================================

    metadata = {
        "schema_version": "feature2-v1",

        "dataset_path": str(
            DATASET_PATH
        ),

        "model_path": str(
            MODEL_PATH
        ),

        "device": str(
            DEVICE
        ),

        "num_samples": int(
            X.shape[0]
        ),

        "num_features": int(
            X.shape[1]
        ),

        "num_train_samples": int(
            len(train_indices)
        ),

        "num_validation_samples": int(
            len(validation_indices)
        ),

        "num_test_samples": int(
            len(test_indices)
        ),

        "feature_names": feature_names.tolist(),

        "target_names": TARGET_NAMES,

        "target_indices": TARGET_INDICES,

        "random_seed": RANDOM_SEED,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "max_epochs": MAX_EPOCHS,

        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "hidden_dim": HIDDEN_DIM,

        "dropout": DROPOUT,

        "best_epoch": int(
            checkpoint["epoch"]
        ),

        "best_validation_loss": float(
            checkpoint[
                "best_validation_loss"
            ]
        ),

        "test_loss": float(
            test_loss
        ),

        "test_metrics": test_metrics,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    logger.info(
        "Training metadata saved: %s",
        METADATA_PATH,
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    logger.info(
        ""
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "FEATURE 2 TRAINING COMPLETE"
    )

    logger.info(
        "=" * 72
    )

    logger.info(
        "Best model : %s",
        MODEL_PATH,
    )

    logger.info(
        "Scaler     : %s",
        SCALER_PATH,
    )

    logger.info(
        "Splits     : %s",
        SPLITS_PATH,
    )

    logger.info(
        "Metadata   : %s",
        METADATA_PATH,
    )

    logger.info(
        "History    : %s",
        HISTORY_PATH,
    )

    logger.info(
        "=" * 72
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()