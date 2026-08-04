# BirdCLEF 2026 — 12th Place Solution

Audio classification system for the [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026) competition.

**12th place out of 4,094 teams** · **Gold medal** · **0.960 public / 0.956 private leaderboard AUC**

## Context

BirdCLEF is a multi-label bioacoustic classification challenge: given long, noisy soundscape recordings, the goal is to identify birds and other sound-producing species. This project was an opportunity to explore robust learning under sparse and imperfect labels, limited examples, background species, and domain shift between focal recordings and soundscapes.

The final solution combined EfficientNetV2 audio classifiers, Xeno-Canto pre-training, class-aware asymmetric loss, semi-supervised learning, and a consistency regularizer developed for this project.

## Approach

### Data

- Recovered **369 corrupted recordings** from their original sources.
- Removed hash duplicates to prevent leakage; **86 unique recordings** appeared as many as three times.
- Manually excluded misleading sections from sparse minority-class recordings.
- Mined verified silent soundscapes as negative training examples.
- Added Xeno-Canto recordings found through taxonomic synonyms for two minority species.
- Pre-trained backbones on a larger Xeno-Canto collection.

### Modeling

The models use EfficientNetV2-B0, B3, and S backbones with two-layer classification heads. They were trained on augmented spectrograms with AdamW and cosine learning-rate decay.

Two main ideas drove the solution:

- **Class-wise Asymmetric Loss (CASL):** class-dependent focusing parameters softly down-weight labels more likely to be noisy.
- **MixMax consistency:** recordings are mixed while their targets are combined using a maximum rather than linear interpolation. Predictions are encouraged to remain consistent across the original recordings, their mixture, and different augmentations. This provides both regularization and a self-supervised signal when labels are incomplete.

The baseline was extended with online pseudo-labeling over unlabeled soundscapes. Fold-specific teacher predictions were sharpened and averaged before supervising the student models.

### Post-processing

The final pipeline used:

- logit sharpening;
- 2.5-second sliding-window test-time augmentation;
- temporal smoothing for amphibians and insects;
- file-level maximum smoothing;
- approximate site, hour, and month priors.

Sliding-window inference produced the largest post-processing gain, improving private leaderboard AUC by approximately **0.010**.

## Results

| Submission | Public AUC | Private AUC |
| --- | ---: | ---: |
| Raw ensemble | 0.9476 | 0.9413 |
| Final ensemble | **0.960** | **0.956** |

Xeno-Canto pre-training improved both leaderboard splits by roughly 0.01. CASL and MixMax were modest independently but performed best together, reaching 0.940 private AUC in the baseline ablation before final ensembling and post-processing.

## Project structure

```text
training/           Baseline, semi-supervised, and Xeno-Canto pre-training pipelines
models/             Model definitions and supporting components
pretrained_models/  Pre-trained model assets
teachers/           Teacher models used for pseudo-labeling
deliveries/         Compiled inference deliverables
error_analyses/     Error-analysis utilities and artifacts
labs/               Experiments and exploratory work
writeup.md          Detailed competition solution write-up
```

Key entry points include:

- `training/baseline.py` — supervised training with CASL and MixMax consistency;
- `training/semi_supervised.py` — online pseudo-label training using fold teachers;
- `training/xc_pretraining.py` — backbone pre-training on Xeno-Canto data.

## Resources

- [Detailed solution write-up](writeup.md)
- [Xeno-Canto pre-training metadata](https://www.kaggle.com/datasets/antoinemasq/xeno-canto-pretraining-data/)
- [Bioacoustic annotation tool](https://www.kaggle.com/code/antoinemasq/birdclef-2026-annotation-tool-dash-app)

## Acknowledgements

Thanks to the BirdCLEF organizers and community for creating a competition that made it possible to test, fail, iterate, and learn.
