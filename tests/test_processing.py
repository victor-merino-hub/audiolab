import numpy as np
from scipy import signal
import pytest

from audiolab.analysis import compute_spectrum
from audiolab.generator import generate_signal
from audiolab.processing import add_noise, change_speed, pink_noise, time_stretch


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


def test_pink_noise_has_equal_energy_per_octave():
    fs = 16000
    noise = pink_noise(fs * 10, rng=0)
    f, P = signal.welch(noise, fs, nperseg=4096)
    band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum()
    # Octaves 100-200 Hz and 2-4 kHz: same energy (white noise would give 20x more in the second)
    assert band(2000, 4000) / band(100, 200) == pytest.approx(1, abs=0.2)
    assert np.std(noise) == pytest.approx(1)


@pytest.mark.parametrize("snr_db", [0, 20, 40])
def test_add_noise_snr_ignores_digital_silence(snr_db):
    fs = 8000
    _, tone = generate_signal("sine", 440, 0.5, fs, 1.0)
    y = np.concatenate([tone, np.zeros(fs)])          # half of the clip is padding
    noisy = add_noise(y, snr_db, rng=0)

    noise = noisy - y
    measured = 10 * np.log10(np.mean(tone ** 2) / np.mean(noise ** 2))
    assert measured == pytest.approx(snr_db, abs=0.1)
    assert np.all(noisy[fs:] != 0)                    # no exact zeros left
