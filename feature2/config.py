from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"
ANNOTATION_DIR = DATA_DIR / "annotations"

CLASSICAL_FEATURE_DIR = FEATURE_DIR / "classical"
CLAP_FEATURE_DIR = FEATURE_DIR / "clap"

PROCESSED_DIR = DATA_DIR / "processed" / "feature2"
MODEL_DIR = PROJECT_ROOT / "models" / "feature2"

AROUSAL_FILENAME = "arousal.csv"
VALENCE_FILENAME = "valence.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Model/training defaults
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12

CLAP_DIMENSION = 512

# We will store the exact feature schema alongside the trained model.
SCHEMA_VERSION = "feature2-v1"