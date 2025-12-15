import os
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from load_config import load_config
cfg = load_config("config.yaml")

"""
According to https://zenodo.org/records/15389653, assumes files in DATA_FOLDER follow the pattern:
s[a]_[pedal]_g[x]_t[y].wav
e.g. s0_tube_screamer_g0_t3.wav
"""


DATA_FOLDER = cfg["paths"]["data_folder"]
OUTPUT_FOLDER = cfg["paths"]["data_processed_folder"]
SWEEPS_DIR = OUTPUT_FOLDER / "sweeps"
CHUNKS_DIR = OUTPUT_FOLDER / "audio_chunks"

SWEEP_SR = cfg["data_processing"]["sweep_sr"]
INITIAL_OFFSET = cfg["data_processing"]["sweep_offset"]
SWEEP_LENGTH = cfg["data_processing"]["sweep_length"]
CHUNK_LENGTH = cfg["data_processing"]["chunk_length"]
FILE_PERCENTAGE = cfg["data_processing"]["file_percentage"]

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


SWEEP_PATTERN = re.compile(r"s\d+_([^_]+)_g(\d+)_t(\d+)\.wav$")
CHUNK_PATTERN = re.compile(r"a_([^_]+)_g(\d+)_t(\d+)\.wav$")
def parse_filename(file_name, pattern):
    m = pattern.match(file_name)
    if m is None:
        return None
    pedal_name = m.group(1)
    gain = int(m.group(2))
    tone = int(m.group(3))
    return pedal_name, gain, tone


def extract_sweeps():
    rows = []
    for root, _, files in os.walk(DATA_FOLDER):
        root = Path(root)
        if root == DATA_FOLDER:
            continue

        for file_name in files:
            if not (file_name.startswith("s0") and file_name.endswith(".wav")):
                continue

            parsed = parse_filename(file_name, SWEEP_PATTERN)
            if parsed is None:
                continue

            pedal_name, gain, tone = parsed
            wav_path = root / file_name
            print(f"Processing {wav_path}...")

            audio, _ = librosa.load(wav_path, sr=SWEEP_SR, mono=True)
            sweep = audio[INITIAL_OFFSET:INITIAL_OFFSET + SWEEP_LENGTH]

            npy_name = file_name.replace(".wav", ".npy")
            npy_path = SWEEPS_DIR / npy_name
            np.save(npy_path, sweep.astype(np.float32))

            rows.append(
                {
                    "file": file_name,
                    "pedal": pedal_name,
                    "gain": gain,
                    "tone": tone,
                    "sweep_path": str(npy_path),
                }
            )

    df = pd.DataFrame(rows)
    metadata_path = OUTPUT_FOLDER / "sweeps_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved sweep metadata to: {metadata_path}")


def extract_chunks():
    rows = []
    for root, dirs, files in os.walk(DATA_FOLDER):
        root = Path(root)
        if root == DATA_FOLDER:
            continue

        for file_name in files:
            if not (file_name.startswith("a") and file_name.endswith(".wav")):
                continue

            parsed = parse_filename(file_name, CHUNK_PATTERN)
            if parsed is None:
                continue

            pedal_name, gain, tone = parsed
            wav_path = root / file_name
            print(f"Chunking {wav_path}...")

            audio, _ = librosa.load(wav_path, sr=SWEEP_SR, mono=True)

            total_samples = len(audio)
            max_samples = int(total_samples * FILE_PERCENTAGE)
            if max_samples < CHUNK_LENGTH:
                continue

            max_samples = (max_samples // CHUNK_LENGTH) * CHUNK_LENGTH
            n_chunks = max_samples // CHUNK_LENGTH

            for idx in range(n_chunks):
                start = idx * CHUNK_LENGTH
                end = start + CHUNK_LENGTH
                chunk = audio[start:end]

                chunk_name = f"{pedal_name}_{idx}_g{gain}_t{tone}.npy"
                chunk_path = CHUNKS_DIR / chunk_name
                np.save(chunk_path, chunk.astype(np.float32))

                rows.append(
                    {
                        "file": chunk_name,
                        "pedal": pedal_name,
                        "gain": gain,
                        "tone": tone,
                        "index": idx,
                        "chunk_path": str(chunk_path),
                    }
                )

    df = pd.DataFrame(rows)
    metadata_path = OUTPUT_FOLDER / "audio_chunks_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved chunk metadata to: {metadata_path}")


if __name__ == "__main__":
    extract_sweeps()
    extract_chunks()