"""Neural network architecture (CNN) to classify spectrograms."""
import torch.nn as nn


class AudioCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 1 channel -> 8 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # halves the size

            nn.Conv2d(8, 16, kernel_size=3, padding=1),   # 8 filters -> 16
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # average each activation map down to 1 number
            nn.Flatten(),               # (batch, 16, 1, 1) -> (batch, 16)
            nn.Linear(16, n_classes),   # 16 -> 4 scores (one per class)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)
