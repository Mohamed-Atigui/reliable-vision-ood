from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataLoaders:
    train: object
    calibration: object
    test: object
    ood: object


def build_loaders(data_dir: str, batch_size: int, num_workers: int, calibration_fraction: float, seed: int):
    """Create disjoint CIFAR-10 train/calibration/test splits and an SVHN OOD loader."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    augmented = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    evaluation = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)
    ood = datasets.SVHN(data_dir, split="test", download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(augmented), generator=generator).tolist()
    n_calibration = int(len(indices) * calibration_fraction)
    calibration_indices, train_indices = indices[:n_calibration], indices[n_calibration:]
    options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return DataLoaders(
        train=DataLoader(Subset(augmented, train_indices), shuffle=True, **options),
        calibration=DataLoader(Subset(evaluation, calibration_indices), shuffle=False, **options),
        test=DataLoader(test, shuffle=False, **options),
        ood=DataLoader(ood, shuffle=False, **options),
    )
