import numpy as np
import pytest
from scipy import signal

from audiolab.analysis import compute_spectrum, compute_spectrogram
from audiolab.generator import generate_signal


def test_spectrum_of_sine():
    # 1 s at 8 kHz -> 1 Hz resolution, so 440 Hz falls exactly on a bin
    fs, A = 8000, 0.8
    _, y = generate_signal("sine", 440, A, fs, 1.0)
    f, P = compute_spectrum(y, fs)

    # Two-sided spectrum: the energy splits into +440 Hz and -440 Hz, A/2 each
    peaks = np.sort(f[np.argsort(P)[-2:]])
    assert np.allclose(peaks, [-440, 440])
    assert P.max() == pytest.approx(A / 2, rel=1e-3)


def test_spectrum_is_symmetric_for_real_signals():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    f, P = compute_spectrum(y, fs=1000)
    # |Y(f)| = |Y(-f)|; the first bin (-fs/2) has no positive counterpart
    assert np.allclose(P[1:], P[1:][::-1])


def test_spectrogram_tracks_a_chirp():
    fs = 16000
    t = np.arange(0, 2.0, 1 / fs)
    y = signal.chirp(t, f0=500, t1=2.0, f1=5000)
    f, tt, S_dB = compute_spectrogram(y, fs)

    assert S_dB.shape == (len(f), len(tt))
    dominant = f[np.argmax(S_dB, axis=0)]           # strongest frequency in each frame
    assert dominant[0] < 1000 and dominant[-1] > 4500
    assert np.all(np.diff(dominant) >= 0)           # the frequency only goes up
