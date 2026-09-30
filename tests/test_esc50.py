import numpy as np
import pandas as pd

from audiolab.audio_io import save_audio
from audiolab.esc50 import load_clip, load_metadata
from audiolab.generator import generate_signal


def make_fake_esc50(root):
    """A tiny dataset with the ESC-50 layout, so the tests do not need the download."""
    (root / "meta").mkdir(parents=True)
    (root / "audio").mkdir()
    rows = [("1-100-A-0.wav", 1, 0, "dog"), ("2-200-A-42.wav", 2, 42, "siren")]
    pd.DataFrame(rows, columns=["filename", "fold", "target", "category"]).to_csv(
        root / "meta" / "esc50.csv", index=False
    )
    for filename, *_ in rows:
        _, y = generate_signal("sine", 440, 0.5, 44100, 0.1)
        save_audio(root / "audio" / filename, y, 44100)


def test_metadata_and_groups(tmp_path):
    make_fake_esc50(tmp_path)
    meta = load_metadata(tmp_path)
    assert len(meta) == 2
    assert list(meta.group) == ["animals", "urban"]    # targets 0-9 and 40-49


def test_load_clip(tmp_path):
    make_fake_esc50(tmp_path)
    y, fs = load_clip(tmp_path, "1-100-A-0.wav")
    assert fs == 44100
    assert len(y) == 4410
    assert np.abs(y).max() > 0
