"""Audio feature extraction for ML."""
import numpy as np
import librosa


def extract_mfcc_features(y, fs, n_mfcc=20):
    """Summarize a signal as a fixed vector of 2*n_mfcc numbers (mean and std of the MFCCs)."""
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)   # matrix (n_mfcc, frames)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_mfcc_features_dataset(X, fs, n_mfcc=20):
    """Apply extract_mfcc_features to each row of X. Returns a matrix (n_signals, 2*n_mfcc)."""
    return np.array([extract_mfcc_features(sig, fs, n_mfcc) for sig in X])


def extract_harmonic_features(y, fs, n_harm=8):
    """Features based on the harmonic structure.

    Returns a vector of n_harm numbers:
        [f0 normalized by Nyquist, relative amplitude of harmonics 2..n_harm]
    """
    N = len(y)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(N)))     # spectrum with a Hann window
    resolution = fs / N                                    # Hz per spectrum bin

    # 1. Fundamental frequency: the highest peak (skipping the 0 Hz bin)
    i0 = np.argmax(spectrum[1:]) + 1
    f0 = i0 * resolution
    a1 = spectrum[i0]

    # 2. Amplitude of each harmonic relative to the fundamental
    ratios = []
    for k in range(2, n_harm + 1):
        fk = k * f0
        if fk >= fs / 2 - 3 * resolution:
            ratios.append(0.0)                             # the harmonic falls outside the spectrum
        else:
            j = int(round(fk / resolution))
            ratios.append(spectrum[j - 2:j + 3].max() / a1)

    return np.array([f0 / (fs / 2)] + ratios)


def extract_harmonic_features_dataset(X, fs, n_harm=8):
    """Apply extract_harmonic_features to each row of X."""
    return np.array([extract_harmonic_features(sig, fs, n_harm) for sig in X])
