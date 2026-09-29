"""Generación de datasets sintéticos etiquetados para ML."""
import numpy as np
from audiolab.generador import generar_senal

TIPOS = ["seno", "cuadrada", "triangular", "sierra"]


def generar_dataset(n_por_clase=200, fs=16000, duracion=1.0, seed=0, snr_db=None):
    """Genera señales aleatorias etiquetadas por tipo.

    Si snr_db es None, el ruido tiene una intensidad aleatoria (0 a 0.1).
    Si se indica snr_db, el ruido se ajusta a esa relación señal-ruido en dB.

    Devuelve:
        X: array (n_señales, n_muestras)
        y: array de etiquetas (texto), una por señal
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for tipo in TIPOS:
        for _ in range(n_por_clase):
            frecuencia = rng.uniform(100, 2000)
            amplitud = rng.uniform(0.2, 1.0)
            _, senal = generar_senal(tipo, frecuencia, amplitud, fs, duracion)

            if snr_db is None:
                sigma = rng.uniform(0, 0.1)
            else:
                potencia_senal = np.mean(senal ** 2)
                sigma = np.sqrt(potencia_senal / 10 ** (snr_db / 10))

            senal = senal + rng.normal(0, sigma, size=senal.shape)
            X.append(senal)
            y.append(tipo)
    return np.array(X), np.array(y)