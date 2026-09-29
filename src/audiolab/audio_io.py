"""Entrada/salida de audio: cargar, guardar y grabar."""
import librosa
import soundfile as sf
import sounddevice as sd


def cargar_audio(ruta):
    """Carga un archivo de audio en mono, conservando su frecuencia de muestreo."""
    y, fs = librosa.load(ruta, sr=None, mono=True)
    return y, fs


def guardar_audio(ruta, y, fs):
    """Guarda una señal en un archivo WAV."""
    sf.write(ruta, y, fs)


def grabar_audio(duracion, fs=44100):
    """Graba con el micrófono por defecto durante `duracion` segundos (mono)."""
    y = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()                      # espera a que termine la grabación
    return y[:, 0], fs