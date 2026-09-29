import numpy as np
import pytest

from audiolab.analysis import compute_spectrum
from audiolab.generator import generate_signal
from audiolab.processing import change_speed, time_stretch


def peak_frequency(y, fs):
    f, P = compute_spectrum(y, fs)
    pos = f > 0
    return f[pos][np.argmax(P[pos])]


def test_change_speed_raises_pitch():
    fs = 8000
    _, y = generate_signal("sine", 440, 0.5, fs, 1.0)
    y2, fs2 = change_speed(y, fs, 2)

    assert fs2 == 2 * fs
    assert len(y2) / fs2 == pytest.approx(0.5)              # half the duration
    assert peak_frequency(y2, fs2) == pytest.approx(880)    # one octave up


def test_time_stretch_keeps_pitch():
    fs = 8000
    _, y = generate_signal("sine", 440, 0.5, fs, 1.0)
    y2, fs2 = time_stretch(y, fs, 2)

    assert fs2 == fs
    assert len(y2) / fs2 == pytest.approx(0.5, abs=0.01)
    assert peak_frequency(y2, fs2) == pytest.approx(440, abs=5)
