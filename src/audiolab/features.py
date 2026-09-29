"""Extracción de características (features) de audio para ML."""
import numpy as np
import librosa


def extraer_features(y, fs, n_mfcc=20):
    """Resume una señal en un vector fijo de 2*n_mfcc números (media y desviación de los MFCC)."""
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)   # matriz (n_mfcc, tramos)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extraer_features_dataset(X, fs, n_mfcc=20):
    """Aplica extraer_features a cada fila de X. Devuelve matriz (n_señales, 2*n_mfcc)."""
    return np.array([extraer_features(senal, fs, n_mfcc) for senal in X])


def extraer_features_armonicos(y, fs, n_arm=8):
    """Features basadas en la estructura de armónicos.

    Devuelve un vector de n_arm números:
        [f0 normalizada por Nyquist, amplitud relativa de los armónicos 2..n_arm]
    """
    N = len(y)
    espectro = np.abs(np.fft.rfft(y * np.hanning(N)))     # espectro con ventana Hann
    resolucion = fs / N                                    # Hz por muestra del espectro

    # 1. Frecuencia fundamental: el pico más alto (ignoramos la muestra de 0 Hz)
    i0 = np.argmax(espectro[1:]) + 1
    f0 = i0 * resolucion
    a1 = espectro[i0]

    # 2. Amplitud de cada armónico relativa a la fundamental
    ratios = []
    for k in range(2, n_arm + 1):
        fk = k * f0
        if fk >= fs / 2 - 3 * resolucion:
            ratios.append(0.0)                             # el armónico se sale del espectro
        else:
            j = int(round(fk / resolucion))
            ratios.append(espectro[j - 2:j + 3].max() / a1)

    return np.array([f0 / (fs / 2)] + ratios)


def extraer_features_armonicos_dataset(X, fs, n_arm=8):
    """Aplica extraer_features_armonicos a cada fila de X."""
    return np.array([extraer_features_armonicos(senal, fs, n_arm) for senal in X])