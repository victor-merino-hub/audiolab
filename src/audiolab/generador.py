"""Periodic signals generator (sine, cosine, square, triangular, saw)."""
import numpy as np
from scipy import signal


def generar_senal(tipo, frecuencia, amplitud, fs, duracion):
    """Genera una señal periódica.

    Parámetros:
        tipo: "seno", "coseno", "cuadrada", "triangular" o "sierra"
        frecuencia: frecuencia de la señal en Hz
        amplitud: amplitud máxima (>0)
        fs: frecuencia de muestreo en Hz
        duracion: duración en segundos

    Devuelve:
        t: vector de tiempo (s)
        y: señal generada
    """
    if frecuencia <= 0 or amplitud <= 0 or fs <= 0 or duracion <= 0:
        raise ValueError("Todos los parámetros deben ser mayores que 0")
    if frecuencia > fs / 2:
        raise ValueError("La frecuencia supera Fs/2: habría aliasing")

    n = int(round(duracion * fs))
    t = np.arange(n) / fs
    w = 2 * np.pi * frecuencia * t

    if tipo == "seno":
        y = np.sin(w)
    elif tipo == "coseno":
        y = np.cos(w)
    elif tipo == "cuadrada":
        y = signal.square(w)
    elif tipo == "triangular":
        y = signal.sawtooth(w, 0.5)
    elif tipo == "sierra":
        y = signal.sawtooth(w)
    else:
        raise ValueError(f"Tipo desconocido: {tipo}")

    return t, amplitud * y