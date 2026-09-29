"""Preparación de datos para redes neuronales (PyTorch)."""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from audiolab.analisis import calcular_espectrograma
from audiolab.dataset import TIPOS


def preparar_datos(X, y, fs, test_size=0.2, batch_size=32, seed=0):
    """Convierte señales y etiquetas en DataLoaders de PyTorch.

    Devuelve: train_loader, test_loader, (media, std) usadas para normalizar.
    """
    # 1. Cada señal -> espectrograma en dB
    S = np.array([calcular_espectrograma(senal, fs)[2] for senal in X])   # (n, frecuencias, tiempos)

    # 2. Etiquetas de texto -> números (0, 1, 2, 3)
    y_num = np.array([TIPOS.index(etiqueta) for etiqueta in y])

    # 3. Separar entrenamiento y prueba (antes de normalizar)
    S_train, S_test, y_train, y_test = train_test_split(
        S, y_num, test_size=test_size, stratify=y_num, random_state=seed
    )

    # 4. Normalizar con la media y desviación del ENTRENAMIENTO
    media, std = S_train.mean(), S_train.std()
    S_train = (S_train - media) / std
    S_test = (S_test - media) / std

    # 5. A tensores de PyTorch con forma (n, 1, frecuencias, tiempos)
    def a_tensor(S):
        return torch.tensor(S, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(a_tensor(S_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(a_tensor(S_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, (media, std)