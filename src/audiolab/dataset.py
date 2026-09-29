"""Labeled synthetic datasets for ML."""
import numpy as np
from audiolab.generator import generate_signal

WAVEFORMS = ["sine", "square", "triangle", "sawtooth"]


def generate_dataset(n_per_class=200, fs=16000, duration=1.0, seed=0, snr_db=None):
    """Generate random signals labeled by waveform.

    If snr_db is None, the noise has a random intensity (0 to 0.1).
    If snr_db is given, the noise is scaled to that signal-to-noise ratio in dB.

    Returns:
        X: array (n_signals, n_samples)
        y: array of labels (strings), one per signal
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for waveform in WAVEFORMS:
        for _ in range(n_per_class):
            frequency = rng.uniform(100, 2000)
            amplitude = rng.uniform(0.2, 1.0)
            _, sig = generate_signal(waveform, frequency, amplitude, fs, duration)

            if snr_db is None:
                sigma = rng.uniform(0, 0.1)
            else:
                signal_power = np.mean(sig ** 2)
                sigma = np.sqrt(signal_power / 10 ** (snr_db / 10))

            sig = sig + rng.normal(0, sigma, size=sig.shape)
            X.append(sig)
            y.append(waveform)
    return np.array(X), np.array(y)
