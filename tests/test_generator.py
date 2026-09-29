import numpy as np
import pytest

from audiolab.generator import generate_signal

WAVEFORMS = ["sine", "cosine", "square", "triangle", "sawtooth"]


def test_time_vector():
    t, y = generate_signal("sine", 100, 1.0, fs=8000, duration=0.5)
    assert len(t) == len(y) == 4000
    assert t[0] == 0
    assert np.allclose(np.diff(t), 1 / 8000)


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_peak_amplitude(waveform):
    _, y = generate_signal(waveform, 100, 0.7, fs=8000, duration=0.1)
    assert np.abs(y).max() == pytest.approx(0.7, abs=1e-3)


def test_sine_and_cosine_initial_values():
    _, s = generate_signal("sine", 100, 2.0, fs=8000, duration=0.01)
    _, c = generate_signal("cosine", 100, 2.0, fs=8000, duration=0.01)
    assert s[0] == pytest.approx(0)
    assert c[0] == pytest.approx(2.0)


@pytest.mark.parametrize("args", [
    ("sine", -100, 1.0, 8000, 1.0),     # negative frequency
    ("sine", 100, 0, 8000, 1.0),        # zero amplitude
    ("sine", 5000, 1.0, 8000, 1.0),     # above Nyquist -> aliasing
    ("noise", 100, 1.0, 8000, 1.0),     # unknown waveform
])
def test_invalid_arguments(args):
    with pytest.raises(ValueError):
        generate_signal(*args)
