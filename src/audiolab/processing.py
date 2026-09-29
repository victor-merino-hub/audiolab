"""Signal processing: speed change and time-stretch."""
import librosa


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
