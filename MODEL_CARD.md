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

## Limitations

- SVHN is a clear semantic shift and does not represent every real distribution shift.
- ECE depends on binning and should not be interpreted alone.
- Split conformal coverage is marginal and relies on exchangeability between calibration and evaluation examples.
- The abstention mechanism has no application-specific error cost.
- Results depend on hardware, seed and training budget; the repository records these settings explicitly.
