# Handwritten Digit Recognition (MNIST)

A PyTorch CNN that classifies handwritten digits (0–9) from the MNIST dataset — built as a foundational project to learn the core PyTorch training pipeline: `Dataset`/`DataLoader`, CNN architecture, the training loop (forward pass, loss, backpropagation, optimizer step), and evaluation.

## Overview

- **Dataset**: MNIST — 70,000 grayscale images (28×28), digits 0–9
- **Model**: Convolutional Neural Network (CNN) — convolutional layers for feature extraction, pooling for downsampling, fully connected layers for classification
- **Framework**: PyTorch, trained locally with Apple Silicon (MPS) acceleration
- **Result**: [X]% test accuracy

## What this project covers

- Loading and batching image data with `Dataset` and `DataLoader`
- Building a CNN from scratch with `nn.Module`
- Training loop: forward pass → cross-entropy loss → backpropagation → optimizer step
- Model evaluation on a held-out test set (accuracy, confusion matrix)

## How to run

```bash
pip install torch torchvision
python train.py
```

## Notes

This is a foundational learning project, built to establish PyTorch fundamentals before moving to more complex computer vision work.
