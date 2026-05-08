import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Small residual block used in MiniResNet."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class MiniResNet(nn.Module):
    """
    Alternative MiniResNet model for Devanagari digit classification.

    This is not used by default in the notebook. It is added as a stronger
    experimental architecture option next to the baseline CNN.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 64, stride=1, dropout=0.05),
            ResidualBlock(64, 128, stride=2, dropout=0.10),
            ResidualBlock(128, 128, stride=1, dropout=0.10),
            ResidualBlock(128, 256, stride=2, dropout=0.15),
            ResidualBlock(256, 256, stride=1, dropout=0.15),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


def count_trainable_params(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MiniResNet(num_classes=10)
    print(model)
    print(f"Trainable params: {count_trainable_params(model):,}")
