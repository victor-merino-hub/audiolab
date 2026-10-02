import numpy as np
import pytest
import torch

from audiolab.dataset import generate_dataset
from audiolab.generator import generate_signal
from audiolab.processing import colored_noise
from audiolab.features import extract_logmel
from audiolab.deep import (augment_spectrogram, cnn_predict_on_folds, make_noise_augment, mix_noise_db,
                           noise_bank_logmel, predict, predict_with_saved_folds, prepare_data)
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
    assert np.array_equal(predict_with_saved_folds(tmp_path, make_model, S, folds), pred)


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


def test_mix_noise_db_matches_mixing_the_audio():
    # Adding powers in the log-mel domain ~ adding the waveforms (the cross term averages out)
    fs = 22050
    _, tone = generate_signal("sine", 440, 0.5, fs, 2.0)
    noise = colored_noise(len(tone), 1, rng=0) * 0.05
    slow = extract_logmel(tone + noise, fs)
    to_t = lambda A: torch.tensor(A, dtype=torch.float32)[None, None]
    S, N = to_t(extract_logmel(tone, fs)), to_t(extract_logmel(noise, fs))
    snr = 10 * torch.log10((10 ** (S / 10)).sum(2).mean() / (10 ** (N / 10)).sum(2).mean())   # gain = 1
    fast = mix_noise_db(S, N, snr.view(1)).numpy()[0, 0]
    err = slow - fast
    assert np.median(np.abs(err)) < 0.5
    assert abs(np.mean(err)) < 0.5                     # no bias


def test_mix_noise_db_removes_digital_silence():
    S = torch.full((2, 1, 16, 20), -100.0)
    S[:, :, :, :10] = -20.0                            # half sound, half digital silence
    N = torch.full((2, 1, 16, 20), -40.0)
    out = mix_noise_db(S, N, torch.tensor([20.0, 40.0]))
    assert torch.all(out > -99)                        # no exact silence left
    # Requested SNR: noise power 20 / 40 dB below the signal power
    assert out[0, 0, 0, -1].item() == pytest.approx(-40.0, abs=0.1)
    assert out[1, 0, 0, -1].item() == pytest.approx(-60.0, abs=0.1)


def test_noise_augment_probability():
    bank = noise_bank_logmel(n_frames=20, n_per_color=2)
    x = torch.zeros(8, 1, 128, 20)
    assert torch.equal(make_noise_augment(bank, mean=-50, std=10, p=0)(x), x)
    assert not torch.equal(make_noise_augment(bank, mean=-50, std=10, p=1)(x), x)
