import numpy as np
import pytest
import torch

from audiolab.dataset import generate_dataset
from audiolab.generator import generate_signal
from audiolab.features import extract_logmel
from audiolab.deep import augment_spectrogram, cnn_predict_on_folds, predict, prepare_data
from audiolab.model import AudioCNN


@pytest.mark.parametrize("keep_frequency", [True, False])
def test_cnn_output_shape(keep_frequency):
    model = AudioCNN(n_classes=50, n_freq=128, keep_frequency=keep_frequency)
    x = torch.randn(8, 1, 128, 216)      # (batch, channel, mel bands, frames)
    assert model(x).shape == (8, 50)


def test_cnn_ignores_time_shifts():
    # Average over time at the end: a circularly shifted input gives almost the same output
    torch.manual_seed(0)
    model = AudioCNN(n_classes=5, n_freq=64).eval()
    x = torch.randn(1, 1, 64, 128)
    shifted = torch.roll(x, shifts=64, dims=3)
    assert torch.allclose(model(x), model(shifted), atol=0.2)


def test_augmentation_keeps_shape_and_masks():
    x = torch.ones(4, 1, 64, 100)
    out = augment_spectrogram(x, freq_mask=8, time_mask=10)
    assert out.shape == x.shape
    assert torch.all(x == 1)                          # the input is not modified
    for i in range(4):
        assert (out[i] == 0).sum() >= 8 * 100         # at least one full frequency band masked


def test_cnn_pipeline_learns_an_easy_task(tmp_path):
    # End-to-end check (normalization, training, prediction): low tone vs. high tone vs. noise
    fs, n = 8000, 30
    rng = np.random.default_rng(0)
    clips = (
        [generate_signal("sine", rng.uniform(200, 400), 0.5, fs, 0.5)[1] for _ in range(n)]
        + [generate_signal("sine", rng.uniform(1500, 2500), 0.5, fs, 0.5)[1] for _ in range(n)]
        + [rng.normal(0, 0.3, fs // 2) for _ in range(n)]
    )
    S = np.array([extract_logmel(c, fs, n_mels=32, n_fft=256, hop_length=128) for c in clips])
    y = np.repeat([0, 1, 2], n)
    folds = np.tile([1, 2], len(y) // 2)

    make_model = lambda: AudioCNN(n_classes=3, n_freq=32, channels=(8, 16))
    pred, losses = cnn_predict_on_folds(make_model, S, y, folds, verbose=False, epochs=10, lr=3e-3,
                                        save_dir=tmp_path)
    assert losses[1][-1] < losses[1][0]      # the training error goes down
    assert np.mean(pred == y) > 0.9

    # The saved fold model reproduces its predictions
    saved = torch.load(tmp_path / "fold1.pt")
    model = make_model()
    model.load_state_dict(saved["state_dict"])
    X1 = torch.tensor((S[folds == 1] - saved["mean"]) / saved["std"], dtype=torch.float32).unsqueeze(1)
    assert np.array_equal(predict(model, X1), pred[folds == 1])


def test_prepare_data():
    fs = 16000
    X, y = generate_dataset(n_per_class=10, fs=fs, duration=1.0)
    train_loader, test_loader, (mean, std) = prepare_data(X, y, fs, test_size=0.25, batch_size=8)

    assert len(train_loader.dataset) == 30
    assert len(test_loader.dataset) == 10

    xb, yb = next(iter(train_loader))
    assert xb.shape[:2] == (8, 1)
    assert yb.dtype == torch.long
    assert set(yb.tolist()) <= {0, 1, 2, 3}

    # Training data normalized with its own statistics -> mean 0, std 1
    S_train = train_loader.dataset.tensors[0]
    assert S_train.mean().item() == pytest.approx(0, abs=1e-4)
    assert S_train.std().item() == pytest.approx(1, abs=1e-2)
