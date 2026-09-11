# Reliable Vision: Calibration, Conformal Prediction and OOD Detection

An end-to-end PyTorch study of when a CIFAR-10 classifier should be trusted, return several plausible classes,
or abstain. The project separates predictive accuracy from reliability and evaluates a ResNet-18 under both
in-distribution and out-of-distribution data.

## Research questions

1. Does temperature scaling improve probabilistic calibration without changing accuracy?
2. Do adaptive conformal prediction sets reach their nominal marginal coverage?
3. How quickly does classification risk decrease when the model abstains on uncertain examples?
4. Can maximum-softmax-probability and energy scores distinguish CIFAR-10 from SVHN?

## Methodology

- **Classifier:** ResNet-18 adapted to 32x32 inputs and trained from scratch on CIFAR-10.
- **Leakage control:** a seeded calibration subset is removed before training; the test set is untouched.
- **Calibration:** one temperature is selected by minimizing held-out negative log-likelihood.
- **Conformal prediction:** Adaptive Prediction Sets use a finite-sample corrected calibration quantile.
- **Selective prediction:** examples are ordered by calibrated confidence to obtain a risk-coverage curve.
- **OOD detection:** SVHN is evaluated with MSP and energy scores using AUROC, AUPR-Out and FPR@95TPR.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python -m reliable_vision.train --config configs/default.yaml
python -m reliable_vision.evaluate --config configs/default.yaml
```

The data are downloaded automatically by torchvision. A CUDA GPU is recommended for training; CPU execution
is supported but substantially slower. Evaluation writes `artifacts/metrics.json` and
`artifacts/risk_coverage.json`. Checkpoints and datasets are intentionally excluded from Git.

## Evaluation protocol

| Component | Data used | Purpose |
| --- | --- | --- |
| Training | 90% of CIFAR-10 train | Fit ResNet parameters |
| Calibration | 10% of CIFAR-10 train | Fit temperature and APS threshold |
| ID test | CIFAR-10 test | Accuracy, calibration, sets and abstention |
| OOD test | SVHN test | OOD discrimination |

The project reports accuracy, NLL, ECE, Brier score, conformal coverage and set size, plus OOD AUROC,
AUPR-Out and FPR@95TPR.

## Results

The default 30-epoch experiment was run on a Tesla T4 with seed 42. Temperature scaling preserved the
classifier's 93.05% test accuracy while reducing 15-bin ECE from 3.25% to 0.72% and NLL from 0.243 to 0.213.

| Evaluation | Metric | Result |
| --- | --- | ---: |
| CIFAR-10 | Test accuracy | 93.05% |
| Calibration | ECE, before -> after | 3.25% -> 0.72% |
| Calibration | NLL, before -> after | 0.243 -> 0.213 |
| Calibration | Brier, before -> after | 0.108 -> 0.103 |
| APS (90% target) | Empirical coverage | 99.98% |
| APS (90% target) | Mean set size | 5.92 classes |
| SVHN OOD, MSP | AUROC / AUPR-Out | 91.22% / 94.04% |
| SVHN OOD, energy | AUROC / AUPR-Out | 92.11% / 94.52% |

The APS result is deliberately reported with its mean set size: its near-total coverage is conservative rather
than evidence of sharp prediction sets. Machine-readable results are stored in `artifacts/metrics.json`, and the
Colab notebook reproduces training and evaluation from a fresh checkout.

## Repository structure

```text
.
├── configs/default.yaml
├── src/reliable_vision/
│   ├── calibration.py
│   ├── conformal.py
│   ├── data.py
│   ├── evaluate.py
│   ├── evaluation.py
│   ├── model.py
│   ├── ood.py
│   ├── train.py
│   └── utils.py
├── tests/
├── artifacts/metrics.json
├── Reliable_Vision_OOD_Training.ipynb
├── MODEL_CARD.md
└── pyproject.toml
```

## Reproducibility and responsible interpretation

The configuration records the seed, split fraction, optimization parameters and nominal coverage. The model
card documents intended use and limitations. Calibration does not guarantee robustness, conformal coverage is
marginal rather than class-conditional, and one OOD dataset cannot represent all future shifts.

## References

- He et al. (2016), *Deep Residual Learning for Image Recognition*.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- Romano, Sesia and Candes (2020), *Classification with Valid and Adaptive Coverage*.
- Liu et al. (2020), *Energy-based Out-of-distribution Detection*.
