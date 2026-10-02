"""Audio input/output: load, save and record."""
import librosa
import soundfile as sf
import sounddevice as sd


def load_audio(path, sr=None):
    """Load an audio file as mono. sr=None keeps the original sampling rate, otherwise resamples."""
    y, fs = librosa.load(path, sr=sr, mono=True)
    return y, fs


def save_audio(path, y, fs):
    """Save a signal to a WAV file."""
    sf.write(path, y, fs)


def record_audio(duration, fs=44100):
    """Record from the default microphone for `duration` seconds (mono)."""
    y = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()                      # wait until the recording finishes
    return y[:, 0], fs
