# Model card

## Intended use

Research and educational prototype for studying predictive reliability in image classification. It is not
intended for safety-critical or automated decision-making.

## Data and split

- In-distribution data: CIFAR-10.
- Out-of-distribution data: SVHN test images.
- A seeded subset of the CIFAR-10 training data is reserved exclusively for temperature and conformal calibration.
- CIFAR-10 test labels are used only for final evaluation.

## Model and uncertainty methods

- ResNet-18 with a CIFAR-compatible stem.
- Scalar temperature scaling fitted by calibration negative log-likelihood.
- Adaptive Prediction Sets (APS) with a finite-sample conformal quantile.
- Confidence-based abstention summarized by the risk-coverage curve.
- Maximum-softmax-probability and energy OOD scores.

## Reported metrics

Accuracy, negative log-likelihood, 15-bin ECE, Brier score, conformal coverage, mean prediction-set size,
singleton rate, OOD AUROC, AUPR-Out and FPR@95TPR.

## Default experiment results

- CIFAR-10 test accuracy: 0.9305.
- Temperature: 1.4973; ECE decreased from 0.0325 to 0.0072 and NLL from 0.2435 to 0.2130.
- APS at nominal 0.90 coverage: empirical coverage 0.9998, mean set size 5.9218, singleton rate 0.0987.
- SVHN OOD with MSP: AUROC 0.9122, AUPR-Out 0.9404, FPR@95TPR 0.1975.
- SVHN OOD with energy: AUROC 0.9211, AUPR-Out 0.9452, FPR@95TPR 0.2309.

The unusually conservative APS sets are a limitation of this configuration, not a performance claim.

## Limitations

- SVHN is a clear semantic shift and does not represent every real distribution shift.
- ECE depends on binning and should not be interpreted alone.
- Split conformal coverage is marginal and relies on exchangeability between calibration and evaluation examples.
- The abstention mechanism has no application-specific error cost.
- Results depend on hardware, seed and training budget; the repository records these settings explicitly.
