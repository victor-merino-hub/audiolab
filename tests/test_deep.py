import pytest
import torch

from audiolab.dataset import generate_dataset
from audiolab.deep import prepare_data
from audiolab.model import AudioCNN


def test_cnn_output_shape():
    model = AudioCNN(n_classes=4)
    x = torch.randn(8, 1, 513, 30)       # (batch, channel, frequencies, times)
    assert model(x).shape == (8, 4)


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
