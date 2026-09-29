import numpy as np
import pytest

from audiolab.features import (
    extract_harmonic_features,
    extract_harmonic_features_dataset,
    extract_mfcc_features,
    extract_mfcc_features_dataset,
)
from audiolab.generator import generate_signal

FS = 16000
F0 = 200    # 1 s at 16 kHz -> 1 Hz bins, so every harmonic falls exactly on a bin


def harmonics(waveform):
    _, y = generate_signal(waveform, F0, 1.0, FS, 1.0)
    return extract_harmonic_features(y, FS, n_harm=8)


def test_mfcc_shapes():
    _, y = generate_signal("sine", F0, 1.0, FS, 1.0)
    assert extract_mfcc_features(y, FS, n_mfcc=13).shape == (26,)
    X = np.stack([y, y, y])
    assert extract_mfcc_features_dataset(X, FS, n_mfcc=13).shape == (3, 26)


def test_fundamental_frequency():
    feats = harmonics("sine")
    assert feats[0] * FS / 2 == pytest.approx(F0)


# Fourier series: harmonic k relative to the fundamental
#   sine: none | square: 1/k odd only | triangle: 1/k^2 odd only | sawtooth: 1/k all
@pytest.mark.parametrize("waveform, expected", [
    ("sine",     {2: 0,   3: 0,     5: 0}),
    ("square",   {2: 0,   3: 1/3,   5: 1/5}),
    ("triangle", {2: 0,   3: 1/9,   5: 1/25}),
    ("sawtooth", {2: 1/2, 3: 1/3,   5: 1/5}),
])
def test_harmonic_ratios_match_fourier_series(waveform, expected):
    feats = harmonics(waveform)
    for k, ratio in expected.items():
        assert feats[k - 1] == pytest.approx(ratio, abs=0.01), f"harmonic {k}"


def test_harmonics_above_nyquist_are_zero():
    _, y = generate_signal("sawtooth", 3000, 1.0, FS, 1.0)
    feats = extract_harmonic_features(y, FS, n_harm=8)
    assert np.all(feats[3:] == 0)    # harmonics 4..8 are above 8 kHz


def test_harmonic_dataset_shape():
    X = np.stack([generate_signal("sine", F0, 1.0, FS, 1.0)[1]] * 4)
    assert extract_harmonic_features_dataset(X, FS, n_harm=6).shape == (4, 6)
