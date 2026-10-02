"""Data preparation, training and evaluation for neural networks (PyTorch)."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from audiolab.analysis import compute_spectrogram
from audiolab.dataset import WAVEFORMS


def prepare_data(X, y, fs, test_size=0.2, batch_size=32, seed=0):
    """Turn signals and labels into PyTorch DataLoaders.

    Returns: train_loader, test_loader, (mean, std) used for normalization.
    """
    # 1. Each signal -> spectrogram in dB
    S = np.array([compute_spectrogram(sig, fs)[2] for sig in X])   # (n, frequencies, times)

    # 2. String labels -> integers (0, 1, 2, 3)
    y_num = np.array([WAVEFORMS.index(label) for label in y])

    # 3. Train/test split (before normalizing)
    S_train, S_test, y_train, y_test = train_test_split(
        S, y_num, test_size=test_size, stratify=y_num, random_state=seed
    )

    # 4. Normalize with the TRAINING mean and std
    mean, std = S_train.mean(), S_train.std()
    S_train = (S_train - mean) / std
    S_test = (S_test - mean) / std

    # 5. To PyTorch tensors with shape (n, 1, frequencies, times)
    def to_tensor(S):
        return torch.tensor(S, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(to_tensor(S_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(to_tensor(S_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, (mean, std)


def default_device():
    """The GPU if PyTorch can use one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_spectrogram(x, freq_mask=16, time_mask=32):
    """Random variants of a batch (batch, 1, freq, time), so the network cannot memorize clips.

    Each example is shifted circularly in time by a random amount, and gets one random band of
    frequencies and one random stretch of time set to zero (SpecAugment).
    """
    x = x.clone()
    n_freq, n_time = x.shape[2], x.shape[3]
    for i in range(len(x)):
        x[i] = torch.roll(x[i], shifts=int(torch.randint(n_time, ())), dims=2)
        f0 = int(torch.randint(n_freq - freq_mask, ()))
        t0 = int(torch.randint(n_time - time_mask, ()))
        x[i, :, f0:f0 + freq_mask, :] = 0
        x[i, :, :, t0:t0 + time_mask] = 0
    return x


def train_model(model, X, y, epochs=40, batch_size=32, lr=1e-3, weight_decay=1e-3, augment=None, seed=0,
                verbose=False):
    """Train with AdamW and a one-cycle learning rate schedule. Returns the loss of each epoch."""
    start = time.time()
    device = next(model.parameters()).device             # train wherever the model is (CPU or GPU)
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if augment is not None:
                xb = augment(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)     # 1. predict and measure the error
            loss.backward()                   # 2. gradient of the error for every weight
            optimizer.step()                  # 3. move each weight against its gradient
            scheduler.step()
            total += loss.item() * len(xb)
        losses.append(total / len(X))
        if verbose:
            print(f"  epoch {len(losses):2d}/{epochs}: loss {losses[-1]:.3f}  ({time.time() - start:.0f} s)", flush=True)
    return losses


@torch.no_grad()
def predict(model, X, batch_size=128):
    """Predicted class of each example."""
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(xb.to(device)).argmax(dim=1).cpu() for xb in torch.split(X, batch_size)]).numpy()


def cnn_predict_on_folds(make_model, S, y, folds, verbose=True, save_dir=None, device=None, **train_kwargs):
    """Out-of-fold predictions of a CNN, like evaluation.predict_on_folds for scikit-learn models.

    make_model() must return a new, untrained network. S: spectrograms (n, freq, time).
    If save_dir is given, each fold's weights and normalization are saved there as fold{k}.pt.
    device: where to train (default: the GPU if there is one).
    Returns the predictions and the training losses of each fold.
    """
    device = device or default_device()
    pred = np.empty_like(y)
    histories = {}
    for k in np.unique(folds):
        start = time.time()
        test = folds == k
        mean, std = S[~test].mean(), S[~test].std()           # statistics of the TRAINING folds only
        to_tensor = lambda A: torch.tensor((A - mean) / std, dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(int(k))                              # same initial weights on every run
        model = make_model().to(device)
        histories[k] = train_model(model, to_tensor(S[~test]), torch.tensor(y[~test], dtype=torch.long),
                                   seed=int(k), verbose=verbose, **train_kwargs)
        pred[test] = predict(model, to_tensor(S[test]))
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            weights = {name: w.cpu() for name, w in model.state_dict().items()}   # loadable without a GPU
            torch.save({"state_dict": weights, "mean": float(mean), "std": float(std)},
                       Path(save_dir) / f"fold{k}.pt")
        if verbose:
            acc = np.mean(pred[test] == y[test])
            print(f"Fold {k}: {acc:.1%}  (final loss {histories[k][-1]:.2f}, {time.time() - start:.0f} s)")
    return pred, histories


def predict_with_saved_folds(model_dir, make_model, S, folds):
    """Out-of-fold predictions with the models saved by cnn_predict_on_folds(save_dir=model_dir).

    Each sample is classified by the model of its own fold, which never saw it during training,
    so modified versions of the data (noise, filtering...) can be evaluated without retraining.
    """
    pred = np.empty(len(S), dtype=int)
    for k in np.unique(folds):
        saved = torch.load(Path(model_dir) / f"fold{k}.pt")
        model = make_model()
        model.load_state_dict(saved["state_dict"])
        test = folds == k
        X = torch.tensor((S[test] - saved["mean"]) / saved["std"], dtype=torch.float32).unsqueeze(1)
        pred[test] = predict(model, X)
    return pred
