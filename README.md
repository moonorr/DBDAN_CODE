# DBDAN Core Code

This repository provides the core model implementation for the paper:

**Dynamic Bi-domain Discriminator Adversarial Network for EEG Emotion Recognition**

## Files

- `KAN.py`: implementation of the Kolmogorov-Arnold Network (KAN) layers, including B-spline-based learnable activation functions.
- `models.py`: core network modules of DBDAN, including the KAN-based shared feature extractor, multi-branch domain-specific feature extractors, classifiers, and dual-domain discriminators.

## Note

This repository only contains the core model architecture. Training scripts, dataset preprocessing scripts, and raw EEG datasets are not included.

The SEED, SEED-IV, and DEAP datasets should be obtained from their official sources according to their data access policies.

