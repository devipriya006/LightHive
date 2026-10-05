from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"
ANNOTATION_DIR = DATA_DIR / "annotations"
FEATURE_DIR = DATA_DIR / "features"


# ---------------------------------------------------------
# FIND AUDIO FILES
# ---------------------------------------------------------

def find_audio_files():
    """
    Recursively find all MP3/WAV/FLAC audio files
    inside data/raw.
    """

    extensions = ["*.mp3", "*.wav", "*.flac"]

    audio_files = []

    for extension in extensions:
        audio_files.extend(RAW_AUDIO_DIR.rglob(extension))

    return sorted(audio_files)


# ---------------------------------------------------------
# LOAD AUDIO DATASET
# ---------------------------------------------------------

def load_audio_dataset():
    """
    Create a dataframe containing:
        song_id
        audio_path
    """

    audio_files = find_audio_files()

    records = []

    for audio_path in audio_files:

        # Example:
        # 2.mp3 -> 2
        song_id = audio_path.stem

        records.append({
            "song_id": song_id,
            "audio_path": str(audio_path)
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            f"No audio files found inside: {RAW_AUDIO_DIR}"
        )

    # Remove duplicate IDs if any
    df = df.drop_duplicates(subset=["song_id"])

    return df


# ---------------------------------------------------------
# FIND ANNOTATION FILE
# ---------------------------------------------------------

def find_file(filename):
    """
    Search recursively for a specific annotation file.
    """

    matches = list(ANNOTATION_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {ANNOTATION_DIR}"
        )

    return matches[0]


# ---------------------------------------------------------
# LOAD DYNAMIC AROUSAL
# ---------------------------------------------------------

def load_arousal_annotations():

    path = find_file("arousal.csv")

    df = pd.read_csv(path)

    print("\nArousal file:")
    print(path)

    print("\nArousal columns:")
    print(df.columns.tolist())

    return df


# ---------------------------------------------------------
# LOAD DYNAMIC VALENCE
# ---------------------------------------------------------

def load_valence_annotations():

    path = find_file("valence.csv")

    df = pd.read_csv(path)

    print("\nValence file:")
    print(path)

    print("\nValence columns:")
    print(df.columns.tolist())

    return df


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("NEURALSTAGE AI - FEATURE 1 DATASET LOADER")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nSearching for audio files...")

    audio_df = load_audio_dataset()

    print(f"\nAudio files found: {len(audio_df)}")

    print("\nFirst 10 audio files:")
    print(audio_df.head(10).to_string(index=False))

    print("\nLoading annotations...")

    arousal_df = load_arousal_annotations()
    valence_df = load_valence_annotations()

    print("\n" + "=" * 60)
    print("DATASET CHECK COMPLETE")
    print("=" * 60)

    print(f"\nAudio files : {len(audio_df)}")
    print(f"Arousal rows: {len(arousal_df)}")
    print(f"Valence rows: {len(valence_df)}")