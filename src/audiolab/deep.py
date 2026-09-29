"""Data preparation for neural networks (PyTorch)."""
import numpy as np
import torch
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
