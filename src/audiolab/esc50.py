"""ESC-50 dataset: metadata and audio clips.

ESC-50 (Piczak, 2015) has 2,000 environmental sound clips of 5 s in 50 classes,
prearranged into 5 folds so that clips cut from the same original recording
always share a fold. https://github.com/karolpiczak/ESC-50

Expected layout (the unzipped repository):
    root/meta/esc50.csv
    root/audio/*.wav
"""
from pathlib import Path

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


def load_clip(root, filename):
    """Load one clip by its filename. Returns (y, fs)."""
    return load_audio(Path(root) / "audio" / filename)
