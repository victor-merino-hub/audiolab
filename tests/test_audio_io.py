import numpy as np

from audiolab.audio_io import load_audio, save_audio
from audiolab.generator import generate_signal


def test_save_and_load_roundtrip(tmp_path):
    fs = 22050
    _, y = generate_signal("sine", 440, 0.5, fs, 0.5)
    path = tmp_path / "sine.wav"

    save_audio(path, y, fs)
    y2, fs2 = load_audio(path)

    assert fs2 == fs
    assert np.allclose(y, y2, atol=1e-4)
