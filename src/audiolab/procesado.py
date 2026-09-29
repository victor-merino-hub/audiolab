"""Procesado de señales: cambio de velocidad y time-stretch."""
import librosa


def cambiar_velocidad(y, fs, factor):
    """Mismas muestras, otra frecuencia de muestreo (como tu MATLAB).

    factor > 1: más rápido y más agudo; factor < 1: más lento y más grave.
    """
    return y, fs * factor


def time_stretch(y, fs, factor):
    """Cambia la duración SIN cambiar el tono (vocoder de fase).

    factor > 1: más rápido (más corto); factor < 1: más lento (más largo).
    """
    return librosa.effects.time_stretch(y, rate=factor), fs