"""Signal analysis: frequency spectrum and spectrogram."""
import numpy as np
from scipy import signal


def compute_spectrum(y, fs):
    """Return the two-sided spectrum (f in Hz, normalized amplitude)."""
    L = len(y)
    Y = np.fft.fft(y)                                  # Fourier transform
    Y = np.fft.fftshift(Y)                             # negative frequencies on the left
    P = np.abs(Y / L)                                  # magnitude, normalized by L
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))     # frequency axis in Hz
    return f, P


def compute_spectrogram(y, fs):
    """Return the spectrogram in dB (f in Hz, t in s, power in dB)."""
    nperseg = min(1024, len(y))                        # window; shorter if the signal is short
    f, t, Sxx = signal.spectrogram(
        y, fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=1024,
    )
    return f, t, 10 * np.log10(Sxx + 1e-12)            # dB (1e-12 avoids log(0))
