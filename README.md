# Detection of Adversarial Examples in Deep Neural Networks using Feature Squeezing

A PyTorch implementation of the Feature Squeezing defense method for detecting adversarial examples in deep neural networks, based on the paper by Xu, Evans & Qi (NDSS 2020).

## Overview

Deep neural networks are vulnerable to adversarial examples — inputs crafted with imperceptible perturbations that fool a model into misclassifying them. This project implements **Feature Squeezing**, a lightweight detection framework that compares a model's predictions on the original input versus a "squeezed" version. If the predictions differ significantly, the input is flagged as adversarial.

```
Input ──► DNN ──► Prediction₀ ──┐
    │                             ├──► Compare L1 distance ──► Adversarial / Legitimate
    └──► Squeezer ──► DNN ──► Prediction₁ ──┘
```

## How It Works

### 1. Train a CNN Classifier
A 7-layer CNN is trained on MNIST achieving **98.82% accuracy**.

### 2. Generate Adversarial Examples (FGSM Attack)
The Fast Gradient Sign Method perturbs an input image in the direction that maximizes loss:

```
x_adv = x + ε · sign(∇ₓ J(g(x), y))
```

With `ε = 0.15`, the model misclassifies the perturbed image (e.g., predicts `9` instead of `4`).

### 3. Apply Feature Squeezing
Reduce the bit depth of the adversarial image to squeeze out adversarial noise:

```python
def feature_squeeze(image, bit_depth=2):
    levels = 2 ** bit_depth
    return torch.floor(image * levels) / levels
```

### 4. Detect the Attack
Compare predictions on the original vs squeezed image:

```python
if adversarial_pred != squeezed_pred:
    print("Adversarial Attack Detected!")
```

## Results

| Input | Prediction |
|-------|-----------|
| Original image |  4 (correct) |
| Adversarial image (ε=0.15) |  9 (fooled) |
| Squeezed adversarial (2-bit) |  4 (restored) |
| Detection |  Attack Detected |

## Detection Performance (from paper)

| Dataset | Configuration | Detection Rate | FPR | ROC-AUC |
|---------|--------------|---------------|-----|---------|
| MNIST | 1-bit + 2×2 Median | 98.15% | 3.98% | 99.44% |
| CIFAR-10 | 5-bit + 2×2 Median + NLM | 84.53% | 4.93% | 95.74% |
| ImageNet | 5-bit + 2×2 Median + NLM | 85.94% | 8.33% | 94.24% |

## Project Structure

```
adversarial-attack-detection/
│
├── adversial_attack_detection.ipynb   # Main notebook
├── feature_researchPaper.pdf          # Original paper (Xu, Evans & Qi, NDSS 2018)
└── README.md
```

## Notebook Walkthrough

The notebook covers the following stages end to end:

**1. Setup & Data Loading** — Load MNIST dataset, configure GPU/CPU device

**2. CNN Model Definition** — Two conv layers + two FC layers with ReLU and MaxPool

**3. Model Training** — 3 epochs with Adam optimizer and CrossEntropy loss

**4. FGSM Attack** — Generate adversarial examples using gradient sign method

**5. Feature Squeezing** — Apply 2-bit depth reduction to squeeze adversarial noise

**6. Detection Logic** — Compare squeezed vs original predictions to flag attacks

**7. Visualization** — Side-by-side comparison of original, adversarial, and squeezed images

## Requirements

```
torch
torchvision
matplotlib
numpy
```

Install with:

```bash
pip install torch torchvision matplotlib numpy
```

The notebook runs on **Google Colab** with GPU support (recommended) or locally on CPU.

## Quick Start

1. Open `adversial_attack_detection.ipynb` in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook will train the CNN, generate an adversarial example, apply feature squeezing, and display the detection result

## Key Concepts

**Adversarial Examples** — Inputs with imperceptible perturbations that cause misclassification. Crafted using gradient-based methods like FGSM, BIM, CW, and JSMA.

**Feature Squeezing** — Reduces the input feature space by lowering bit depth or applying spatial smoothing, eliminating the adversarial noise while preserving image semantics.

**L1 Detection Score** — Measures the difference between original and squeezed predictions:
`score = ‖g(x) − g(x_squeezed)‖₁`

**Joint Detection** — Combine multiple squeezers and use the maximum L1 score for more robust detection across different attack types.

## Reference

> Weilin Xu, David Evans, Yanjun Qi.
> **Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks.**
> Network and Distributed Systems Security Symposium (NDSS) 2018.
> https://dx.doi.org/10.14722/ndss.2018.23198
