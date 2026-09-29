"""Arquitectura de la red neuronal (CNN) para clasificar espectrogramas."""
import torch.nn as nn


class CNNAudio(nn.Module):
    def __init__(self, n_clases=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1 canal -> 8 filtros
            nn.ReLU(),
            nn.MaxPool2d(2),                              # reduce el tamaño a la mitad

            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # 8 filtros -> 16
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.clasificador = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # promedia cada mapa de activación a 1 número
            nn.Flatten(),              # (lote, 16, 1, 1) -> (lote, 16)
            nn.Linear(16, n_clases),   # 16 -> 4 puntuaciones (una por clase)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.clasificador(x)