"""Periodic signal generation (sine, cosine, square, triangle, sawtooth)."""
import numpy as np
from scipy import signal


def generate_signal(waveform, frequency, amplitude, fs, duration):
    """Generate a periodic signal.

    Parameters:
        waveform: "sine", "cosine", "square", "triangle" or "sawtooth"
        frequency: signal frequency in Hz
        amplitude: peak amplitude (>0)
        fs: sampling frequency in Hz
        duration: duration in seconds

    Returns:
        t: time vector (s)
        y: generated signal
    """
    if frequency <= 0 or amplitude <= 0 or fs <= 0 or duration <= 0:
        raise ValueError("All parameters must be greater than 0")
    if frequency > fs / 2:
        raise ValueError("Frequency exceeds fs/2: it would alias")

    n = int(round(duration * fs))
    t = np.arange(n) / fs
    w = 2 * np.pi * frequency * t

    if waveform == "sine":
        y = np.sin(w)
    elif waveform == "cosine":
        y = np.cos(w)
    elif waveform == "square":
        y = signal.square(w)
    elif waveform == "triangle":
        y = signal.sawtooth(w, 0.5)
    elif waveform == "sawtooth":
        y = signal.sawtooth(w)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    return t, amplitude * y
