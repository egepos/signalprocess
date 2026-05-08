# Hindi / Devanagari Digit Classification

This repository contains a baseline PyTorch CNN notebook for Hindi/Devanagari digit classification.

## Baseline model

The main working notebook is:

```text
train_devanagari_digits.ipynb
```

It trains the original CNN model and produces:

```text
submission.csv
```

## Alternative model: MiniResNet

An additional model implementation is provided in:

```text
models.py
```

This file contains a `MiniResNet` architecture with:

- residual blocks
- batch normalization
- dropout
- adaptive average pooling

## Why MiniResNet is not the default

The MiniResNet architecture was tested as an experimental alternative to the original CNN baseline.

Its validation accuracy was close to the baseline CNN, but it did not provide a clear or consistent improvement over the existing model. For this reason, the original CNN notebook was kept as the stable default implementation, while MiniResNet remains available as an optional model for comparison and future experiments.

## How to test MiniResNet

In `train_devanagari_digits.ipynb`, add:

```python
from models import MiniResNet
```

Then replace the model initialization:

```python
model = DigitCNN(NUM_CLASSES).to(DEVICE)
```

with:

```python
model = MiniResNet(NUM_CLASSES).to(DEVICE)
```

