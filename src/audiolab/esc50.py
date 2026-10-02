"""ESC-50 dataset: metadata and audio clips.

ESC-50 (Piczak, 2015) has 2,000 environmental sound clips of 5 s in 50 classes,
prearranged into 5 folds so that clips cut from the same original recording
always share a fold. https://github.com/karolpiczak/ESC-50

Expected layout (the unzipped repository):
    root/meta/esc50.csv
    root/audio/*.wav
"""
from pathlib import Path

import numpy as np
import pandas as pd

from audiolab.audio_io import load_audio

# The 50 classes come in 5 major groups of 10 consecutive targets
GROUPS = ["animals", "natural", "human", "domestic", "urban"]


def load_metadata(root):
    """Read the metadata table: one row per clip (filename, fold, target, category, ...).

    Adds a "group" column with the major group of each class.
    """
    meta = pd.read_csv(Path(root) / "meta" / "esc50.csv")
    meta["group"] = [GROUPS[t // 10] for t in meta.target]
    return meta


def load_clip(root, filename, sr=None):
    """Load one clip by its filename, optionally resampled to sr. Returns (y, fs)."""
    return load_audio(Path(root) / "audio" / filename, sr=sr)


def compute_features(root, filenames, extract, sr=None, cache=None):
    """Apply extract(y, fs) to every clip and stack the results into one array.

    If `cache` is a .npy path, the array is saved there the first time and loaded afterwards.
    """
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    features = np.array([extract(*load_clip(root, fn, sr=sr)) for fn in filenames])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, features)
    return features
