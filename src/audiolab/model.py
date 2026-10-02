"""Neural network architecture (CNN) to classify spectrograms."""
import torch.nn as nn


def conv_block(c_in, c_out):
    """3x3 convolution -> batch norm -> ReLU -> 2x2 max pooling (halves both axes)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class AudioCNN(nn.Module):
    """CNN for spectrograms with shape (batch, 1, n_freq, n_frames).

    At the end the feature maps are averaged over time, since *when* a sound happens does not
    change its class. With keep_frequency=True the frequency axis is kept, so the classifier still
    knows *where* in frequency each pattern was; with False it is averaged away too.
    """

    def __init__(self, n_classes=50, n_freq=128, channels=(16, 32, 64, 128),
                 keep_frequency=True, dropout=0.3):
        super().__init__()
        sizes = [1, *channels]
        self.conv = nn.Sequential(*[conv_block(sizes[i], sizes[i + 1]) for i in range(len(channels))])
        self.keep_frequency = keep_frequency

        freq_out = n_freq
        for _ in channels:
            freq_out //= 2                       # each block halves the frequency axis
        n_features = channels[-1] * (freq_out if keep_frequency else 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),                 # randomly switch off features while training
            nn.Linear(n_features, n_classes),    # one score per class
        )

    def forward(self, x):
        x = self.conv(x)                         # (batch, channels, freq, time)
        x = x.mean(dim=3)                        # average over time -> (batch, channels, freq)
        if not self.keep_frequency:
            x = x.mean(dim=2, keepdim=True)      # also over frequency -> (batch, channels, 1)
        return self.classifier(x)


def count_parameters(model):
    """Number of trainable weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
