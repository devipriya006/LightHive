from pathlib import Path
import json
import time

import librosa
import numpy as np
import laion_clap
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = ROOT / "data" / "raw" / "MEMD_audio"
FEATURE_DIR = ROOT / "data" / "features"

CLASSICAL_DIR = FEATURE_DIR / "classical"
CLAP_DIR = FEATURE_DIR / "clap"

CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
CLAP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 22050
N_MFCC = 13

# CLAP processes audio at 48 kHz internally.
# Batch size can be reduced to 8 if RAM becomes a problem.
CLAP_BATCH_SIZE = 16


# ============================================================
# CLASSICAL AUDIO FEATURES
# ============================================================

def extract_audio_features(audio_path):

    y, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = len(y) / sr

    # BPM
    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    bpm = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "song_id": int(audio_path.stem) if audio_path.stem.isdigit() else 0,
        "duration_seconds": float(duration),
        "sample_rate": int(sr),

        "bpm": bpm,
        "beat_count": int(len(beat_frames)),

        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),

        "spectral_centroid_mean":
            float(np.mean(spectral_centroid)),

        "spectral_centroid_std":
            float(np.std(spectral_centroid)),

        "spectral_bandwidth_mean":
            float(np.mean(spectral_bandwidth)),

        "spectral_bandwidth_std":
            float(np.std(spectral_bandwidth)),

        "spectral_rolloff_mean":
            float(np.mean(spectral_rolloff)),

        "spectral_rolloff_std":
            float(np.std(spectral_rolloff)),

        "zero_crossing_rate_mean":
            float(np.mean(zcr)),

        "zero_crossing_rate_std":
            float(np.std(zcr)),

        "mfcc_mean":
            np.mean(mfcc, axis=1).tolist(),

        "mfcc_std":
            np.std(mfcc, axis=1).tolist(),
    }


# ============================================================
# CLASSICAL FEATURE EXTRACTION
# ============================================================

def extract_all_classical_features(audio_files):

    print("\n" + "=" * 60)
    print("CLASSICAL AUDIO FEATURE EXTRACTION")
    print("=" * 60)

    success = 0
    failed = 0

    failed_files = []

    for audio_path in tqdm(
        audio_files,
        desc="Classical features"
    ):

        output_file = (
            CLASSICAL_DIR /
            f"{audio_path.stem}.json"
        )

        # Resume support
        if output_file.exists():
            success += 1
            continue

        try:

            features = extract_audio_features(
                audio_path
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    features,
                    f,
                    indent=2
                )

            success += 1

        except Exception as error:

            failed += 1

            failed_files.append({
                "song_id": audio_path.stem,
                "error": str(error)
            })

            print(
                f"\nERROR: {audio_path.name}"
            )
            print(error)

    # Save error log
    if failed_files:

        with open(
            FEATURE_DIR / "classical_errors.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                failed_files,
                f,
                indent=2
            )

    print("\nClassical extraction complete.")
    print(f"Successful: {success}")
    print(f"Failed:     {failed}")


# ============================================================
# CLAP MODEL
# ============================================================

def load_clap_model():

    print("\n" + "=" * 60)
    print("LOADING CLAP MODEL")
    print("=" * 60)

    model = laion_clap.CLAP_Module(
        enable_fusion=False
    )

    model.load_ckpt()

    print("CLAP model loaded.")

    return model


# ============================================================
# CLAP EXTRACTION
# ============================================================

def extract_all_clap_embeddings(
    model,
    audio_files
):

    print("\n" + "=" * 60)
    print("CLAP EMBEDDING EXTRACTION")
    print("=" * 60)

    embedding_files = []

    for audio_path in audio_files:

        output_file = (
            CLAP_DIR /
            f"{audio_path.stem}.npy"
        )

        if not output_file.exists():
            embedding_files.append(audio_path)

    print(
        f"CLAP embeddings remaining: "
        f"{len(embedding_files)}"
    )

    if not embedding_files:
        print("All CLAP embeddings already exist.")
        return

    total_batches = (
        len(embedding_files) +
        CLAP_BATCH_SIZE -
        1
    ) // CLAP_BATCH_SIZE

    failed = []

    for batch_index in range(
        0,
        len(embedding_files),
        CLAP_BATCH_SIZE
    ):

        batch = embedding_files[
            batch_index:
            batch_index + CLAP_BATCH_SIZE
        ]

        batch_number = (
            batch_index // CLAP_BATCH_SIZE
        ) + 1

        print(
            f"\nCLAP batch "
            f"{batch_number}/{total_batches}"
        )

        print(
            f"Files: "
            f"{batch[0].stem} -> "
            f"{batch[-1].stem}"
        )

        try:

            file_list = [
                str(path)
                for path in batch
            ]

            embeddings = (
                model.get_audio_embedding_from_filelist(
                    x=file_list,
                    use_tensor=False
                )
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            print(
                f"Embedding matrix: "
                f"{embeddings.shape}"
            )

            for audio_path, embedding in zip(
                batch,
                embeddings
            ):

                output_file = (
                    CLAP_DIR /
                    f"{audio_path.stem}.npy"
                )

                np.save(
                    output_file,
                    embedding
                )

        except Exception as error:

            print(
                f"\nCLAP BATCH ERROR: "
                f"{error}"
            )

            for audio_path in batch:

                failed.append({
                    "song_id": audio_path.stem,
                    "error": str(error)
                })

    if failed:

        with open(
            CLAP_DIR / "errors.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                failed,
                f,
                indent=2
            )

    print("\nCLAP extraction complete.")

    print(
        f"Failed files: {len(failed)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("NEURALSTAGE AI - FEATURE 1")
    print("AUDIO FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Find audio
    # --------------------------------------------------------

    audio_files = sorted(
        AUDIO_DIR.glob("*.mp3"),
        key=lambda p: int(p.stem)
    )

    print(
        f"\nAudio files found: "
        f"{len(audio_files)}"
    )

    if not audio_files:

        raise RuntimeError(
            f"No MP3 files found in:\n{AUDIO_DIR}"
        )

    # --------------------------------------------------------
    # Classical features
    # --------------------------------------------------------

    extract_all_classical_features(
        audio_files
    )

    # --------------------------------------------------------
    # CLAP
    # --------------------------------------------------------

    clap_model = load_clap_model()

    extract_all_clap_embeddings(
        clap_model,
        audio_files
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("FEATURE 1 COMPLETE")
    print("=" * 60)

    print(
        f"Total time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"\nClassical features:"
        f"\n  {CLASSICAL_DIR}"
    )

    print(
        f"\nCLAP embeddings:"
        f"\n  {CLAP_DIR}"
    )


if __name__ == "__main__":
    main()