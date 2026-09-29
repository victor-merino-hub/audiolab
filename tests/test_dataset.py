import numpy as np
import pytest

from audiolab.dataset import WAVEFORMS, generate_dataset


def test_shapes_and_balanced_classes():
    X, y = generate_dataset(n_per_class=5, fs=8000, duration=0.5)
    assert X.shape == (20, 4000)
    assert y.shape == (20,)
    for waveform in WAVEFORMS:
        assert np.sum(y == waveform) == 5


def test_same_seed_same_data():
    X1, _ = generate_dataset(n_per_class=3, fs=8000, duration=0.1, seed=42)
    X2, _ = generate_dataset(n_per_class=3, fs=8000, duration=0.1, seed=42)
    X3, _ = generate_dataset(n_per_class=3, fs=8000, duration=0.1, seed=7)
    assert np.array_equal(X1, X2)
    assert not np.array_equal(X1, X3)


@pytest.mark.parametrize("snr_db", [0, 10, 20])
def test_requested_snr(snr_db):
    # Same seed -> same clean signals; the difference is exactly the added noise
    kwargs = dict(n_per_class=10, fs=8000, duration=0.5, seed=0)
    X_clean, _ = generate_dataset(snr_db=200, **kwargs)
    X_noisy, _ = generate_dataset(snr_db=snr_db, **kwargs)

    noise = X_noisy - X_clean
    measured = 10 * np.log10(np.mean(X_clean ** 2, axis=1) / np.mean(noise ** 2, axis=1))
    assert measured.mean() == pytest.approx(snr_db, abs=0.2)
