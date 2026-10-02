"""Signal processing: speed change, time-stretch and background noise."""
import librosa
import numpy as np


def change_speed(y, fs, factor):
    """Same samples, different sampling rate.

    factor > 1: faster and higher-pitched; factor < 1: slower and lower-pitched.
    """
    return y, fs * factor


def time_stretch(y, fs, factor):
    """Change the duration WITHOUT changing the pitch (phase vocoder).

    factor > 1: faster (shorter); factor < 1: slower (longer).
    """
    return librosa.effects.time_stretch(y, rate=factor), fs


def pink_noise(n, rng=None):
    """n samples of pink noise (power ~ 1/f: equal energy per octave), with unit standard deviation.

    Many real backgrounds (fans, distant traffic, a room) are closer to pink than to white noise.
    """
    rng = np.random.default_rng(rng)
    spectrum = np.fft.rfft(rng.normal(size=n))
    f = np.arange(len(spectrum))
    f[0] = 1                                      # avoid dividing by zero at DC
    noise = np.fft.irfft(spectrum / np.sqrt(f), n)    # amplitude ~ 1/sqrt(f) -> power ~ 1/f
    return noise / np.std(noise)


def add_noise(y, snr_db, rng=None):
    """Add pink noise at a given SNR (dB).

    The signal power is measured only where y is not exactly zero, so digital-silence padding
    does not lower the reference.
    """
    active = y[y != 0] if np.any(y != 0) else y
    signal_power = np.mean(active ** 2)
    noise = pink_noise(len(y), rng) * np.sqrt(signal_power / 10 ** (snr_db / 10))
    return (y + noise).astype(y.dtype)
