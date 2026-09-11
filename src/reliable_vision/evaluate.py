from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .calibration import fit_temperature, softmax
from .conformal import calibrate_aps, prediction_sets, set_metrics
from .data import build_loaders
from .evaluation import classification_metrics, collect_logits
from .model import build_resnet18
from .ood import energy_score, msp_score, ood_metrics, risk_coverage
from .utils import save_json, seed_everything


def evaluate(config_path: str) -> dict:
    import torch

    config = yaml.safe_load(Path(config_path).read_text())
    seed_everything(config["seed"])
    device = "cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu"
    if config["device"] != "auto":
        device = config["device"]
    loaders = build_loaders(
        config["data_dir"], config["batch_size"], config["num_workers"],
        config["calibration_fraction"], config["seed"],
    )
    checkpoint = torch.load(Path(config["checkpoint_dir"]) / "resnet18_cifar10.pt", map_location=device)
    model = build_resnet18().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    calibration_logits, calibration_labels = collect_logits(model, loaders.calibration, device)
    test_logits, test_labels = collect_logits(model, loaders.test, device)
    ood_logits, _ = collect_logits(model, loaders.ood, device)

    temperature = fit_temperature(calibration_logits, calibration_labels, config["temperature_grid_size"])
    calibration_probability = softmax(calibration_logits, temperature)
    test_probability = softmax(test_logits, temperature)
    aps_threshold = calibrate_aps(calibration_probability, calibration_labels, config["coverage"])
    sets = prediction_sets(test_probability, aps_threshold)
    results = {
        "temperature": temperature,
        "uncalibrated": classification_metrics(softmax(test_logits), test_labels),
        "temperature_scaled": classification_metrics(test_probability, test_labels),
        "conformal_aps": {"nominal_coverage": config["coverage"], "threshold": aps_threshold, **set_metrics(sets, test_labels)},
        "ood_svhn": {
            "msp": ood_metrics(msp_score(test_logits), msp_score(ood_logits)),
            "energy": ood_metrics(energy_score(test_logits), energy_score(ood_logits)),
        },
    }
    output = Path(config["output_dir"])
    save_json(results, output / "metrics.json")
    save_json(risk_coverage(test_probability, test_labels), output / "risk_coverage.json")
    print(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    evaluate(parser.parse_args().config)


if __name__ == "__main__":
    main()
