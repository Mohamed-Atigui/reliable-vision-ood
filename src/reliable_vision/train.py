from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import build_loaders
from .model import build_resnet18
from .utils import save_json, seed_everything


def train(config_path: str) -> None:
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
    model = build_resnet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=0.9,
        weight_decay=config["weight_decay"], nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = correct = seen = 0
        for inputs, labels in loaders.train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_accuracy": correct / seen}
        history.append(row)
        print(row)

    torch.save({"state_dict": model.state_dict(), "config": config}, checkpoint_dir / "resnet18_cifar10.pt")
    save_json({"history": history}, Path(config["output_dir"]) / "training_history.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
