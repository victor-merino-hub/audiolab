"""Real-time classification: keep the latest audio from the microphone and classify it."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from audiolab.features import extract_logmel
from audiolab.model import AudioCNN
from audiolab.processing import pink_noise

MODEL_SR = 22050                        # sampling rate the CNN was trained at


class RingBuffer:
    """Fixed-size circular buffer that always holds the latest n samples of a stream.

    New samples overwrite the oldest ones, so writing never allocates memory or shifts the array,
    which keeps the audio callback fast.
    """

    def __init__(self, n):
        self.data = np.zeros(n, dtype=np.float32)
        self.pos = 0                    # index where the next sample will be written

    def write(self, block):
        """Append a block of samples (1-D array), overwriting the oldest ones."""
        n = len(self.data)
        block = block[-n:]                      # a block longer than the buffer: only its end survives
        k = len(block)
        space = n - self.pos                    # room left before the end of the array
        if k <= space:
            self.data[self.pos:self.pos + k] = block
        else:                                   # wraps around: fill up to the end, the rest from index 0
            self.data[self.pos:] = block[:space]
            self.data[:k - space] = block[space:]
        self.pos = (self.pos + k) % n

    def read_last(self, m):
        """The latest m samples (m <= n) in chronological order, oldest first, as a new array."""
        start = self.pos - m                    # the latest sample is at pos - 1
        if start >= 0:
            return self.data[start:self.pos].copy()          # a slice is a view: copy it
        # Wraps around: a negative start counts from the end of the array (data[-3:] = last 3)
        return np.concatenate([self.data[start:], self.data[:self.pos]])   # concatenate already copies


def _from_checkpoint(saved):
    """(model, mean, std) from a saved dict with the weights and the input normalization."""
    model = AudioCNN()
    model.load_state_dict(saved["state_dict"])
    model.eval()                        # batch norm with its stored statistics, no dropout
    return model, saved["mean"], saved["std"]


def load_models(model_dir, folds=(1,)):
    """Load the fold models saved by deep.cnn_predict_on_folds: a list of (model, mean, std)."""
    return [_from_checkpoint(torch.load(Path(model_dir) / f"fold{k}.pt")) for k in folds]


def load_packaged_model(path):
    """Load a self-contained model file (weights, normalization and class names), like models/*.pt.

    Returns ([(model, mean, std)], class names), ready for class_probabilities.
    """
    saved = torch.load(path)
    return [_from_checkpoint(saved)], list(saved["classes"])


def to_model_rate(y, sr):
    """Resample a signal from sr to the 22.05 kHz of training with a polyphase filter."""
    if sr == MODEL_SR:
        return y
    ratio = Fraction(MODEL_SR, int(sr))                  # 44100 -> 22050 is 1/2
    return resample_poly(y, ratio.numerator, ratio.denominator).astype(np.float32)


@torch.no_grad()
def class_probabilities(models, y):
    """Probability of each class for a window y at 22.05 kHz, averaged over the models (ensemble)."""
    S = extract_logmel(y, MODEL_SR)
    probs = []
    for model, mean, std in models:
        x = torch.tensor((S - mean) / std, dtype=torch.float32)[None, None]   # (1, 1, n_mels, frames)
        probs.append(torch.softmax(model(x), dim=1)[0].numpy())
    return np.mean(probs, axis=0)


class Smoother:
    """First-order IIR low-pass on a sequence of vectors: p_smooth = alpha * p + (1 - alpha) * p_smooth."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None

    def __call__(self, p):
        self.state = p if self.state is None else self.alpha * p + (1 - self.alpha) * self.state
        return self.state


def frame_levels_db(y, n_frame):
    """Power in dB of consecutive frames of n_frame samples (an incomplete last frame is dropped)."""
    n = len(y) // n_frame
    frames = y[:n * n_frame].reshape(n, n_frame)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def add_comfort_noise(y, level_db, rng=None):
    """Add pink noise at level_db dBFS, like the comfort noise a phone plays during the pauses of a call.

    Noise suppressors in the capture chain can turn quiet moments into exact digital zeros, which the
    CNN learned to associate with some classes (the silence shortcut). A low noise floor removes them
    while barely changing loud events (a -60 dBFS floor under a -25 dBFS event is a 35 dB SNR).
    """
    return y + np.float32(10 ** (level_db / 20)) * pink_noise(len(y), rng).astype(np.float32)


def background_level_db(y, sr):
    """Background level of a quiet recording: median power in dB of its 50 ms frames."""
    return np.median(frame_levels_db(y, sr // 20))


class WindowClassifier:
    """What the demo does with every window; shared by the live demo and the offline replay.

    1. Level gate: there is a sound event if some 50 ms frame is margin_db above the background.
    2. Optional comfort noise, only in the model input (the gate keeps looking at the real signal).
    3. Resample to 22.05 kHz, CNN probabilities, smoothed while the event lasts.
    Returns (probabilities, event, 50 ms frame levels in dB).
    """

    def __init__(self, models, sr, background_db, margin_db=10.0, alpha=0.5, comfort_db=None, seed=0):
        self.models, self.sr = models, sr
        self.threshold_db = background_db + margin_db
        self.comfort_db = comfort_db
        self.smoother = Smoother(alpha)
        self.rng = np.random.default_rng(seed)

    def __call__(self, y):
        levels = frame_levels_db(y, self.sr // 20)
        event = levels.max() > self.threshold_db
        x = y if self.comfort_db is None else add_comfort_noise(y, self.comfort_db, self.rng)
        p = class_probabilities(self.models, to_model_rate(x, self.sr))
        if event:
            p = self.smoother(p)
        else:
            self.smoother.state = None  # start fresh at the next event, without old classes lingering
        return p, event, levels


def replay(y, sr, models, window=5.0, hop=0.5, **options):
    """Run the demo offline on a recording of a live session, with the same windows, gate and smoothing.

    As live, the first `window` seconds are the background calibration and the first prediction is made
    when they end. options: margin_db, alpha, comfort_db, seed (see WindowClassifier).
    Returns the probabilities (n_predictions, n_classes) and whether each window had an event.
    """
    n_window, n_hop = int(window * sr), int(hop * sr)
    classify = WindowClassifier(models, sr, background_level_db(y[:n_window], sr), **options)
    probs, events = [], []
    for end in range(n_window, len(y) + 1, n_hop):
        p, event, _ = classify(y[end - n_window:end])
        probs.append(p)
        events.append(event)
    return np.array(probs), np.array(events)
