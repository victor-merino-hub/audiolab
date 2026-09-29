"""Análisis de señales: espectro de frecuencia y espectrograma."""
import numpy as np
from scipy import signal


def calcular_espectro(y, fs):
    """Devuelve el espectro bilateral (f en Hz, amplitud normalizada)."""
    L = len(y)
    Y = np.fft.fft(y)                                  # transformada de Fourier
    Y = np.fft.fftshift(Y)                             # frecuencias negativas a la izquierda
    P = np.abs(Y / L)                                  # módulo, normalizado por L
    f = np.fft.fftshift(np.fft.fftfreq(L, d=1/fs))     # eje de frecuencias en Hz
    return f, P


def calcular_espectrograma(y, fs):
    """Devuelve el espectrograma en dB (f en Hz, t en s, potencia en dB)."""
    nperseg = min(1024, len(y))                        # ventana; más corta si la señal es corta
    f, t, Sxx = signal.spectrogram(
        y, fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=1024,
    )
    return f, t, 10 * np.log10(Sxx + 1e-12)            # dB (el 1e-12 evita log(0))