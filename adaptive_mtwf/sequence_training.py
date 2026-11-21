"""Minimal training and checkpoint helpers for precomputed numerical sequences."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .cnn import CNNConfig, VariableLengthCNN, make_bce_loss
from .data.batching import DEFAULT_MAX_LENGTH, SequenceItem, collate_sequences
from .metrics import ranking_metrics, threshold_metrics


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2023
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    device: str = "cpu"
    max_length: int = DEFAULT_MAX_LENGTH

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be an integer in [0, 2**63).")
        if any(type(size) is not int or size < 1 for size in (self.epochs, self.batch_size, self.max_length)):
            raise ValueError("Epochs, batch size, and maximum length must be positive integers.")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError("Optimizer must be adam or adamw.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")


def resolve_device(name: str) -> torch.device:
    if name not in ("cpu", "mps"):
        raise ValueError("Choose cpu or mps.")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this process.")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for explicit MPS verification.")
    return torch.device(name)


def _tensor_batches(items: Sequence[SequenceItem], order, config: TrainingConfig, device: torch.device):
    for start in range(0, len(order), config.batch_size):
        rows = [items[index] for index in order[start:start + config.batch_size]]
        batch = collate_sequences(rows, max_length=config.max_length)
        yield (torch.from_numpy(batch.inputs).to(device), torch.from_numpy(batch.valid_mask).to(device),
               torch.from_numpy(batch.targets).to(device))


def _finite_loss(loss: Tensor) -> float:
    value = loss.item()
    if not np.isfinite(value):
        raise FloatingPointError("The batch loss is nonfinite.")
    return value


def train_sequences(model_config: CNNConfig, train: Sequence[SequenceItem], validation: Sequence[SequenceItem],
                    config: TrainingConfig = TrainingConfig(), *, pos_weight=None) -> tuple[VariableLengthCNN, list[dict]]:
    """Fit on caller-provided batches and evaluate each epoch using validation only.

    Seeds govern initialization, dropout, and permutations. CPU training uses
    deterministic operations. Reproducibility on MPS depends on the backend.
    """
    if not train or not validation:
        raise ValueError("Nonempty training and validation sequences are required.")
    device = resolve_device(config.device)
    generator = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    model = VariableLengthCNN(model_config).to(device)
    criterion = make_bce_loss(model_config.num_classes, pos_weight=pos_weight, device=device)
    optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        for inputs, mask, targets in _tensor_batches(train, generator.permutation(len(train)), config, device):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, mask), targets)
            train_total += _finite_loss(loss) * targets.shape[0]
            loss.backward()
            optimizer.step()

        model.eval()
        validation_total = 0.0
        probabilities, labels = [], []
        with torch.inference_mode():
            for inputs, mask, targets in _tensor_batches(validation, range(len(validation)), config, device):
                logits = model(inputs, mask)
                validation_total += _finite_loss(criterion(logits, targets)) * targets.shape[0]
                probabilities.append(logits.sigmoid().cpu().numpy())
                labels.append(targets.cpu().numpy())
        scores, truth = np.concatenate(probabilities), np.concatenate(labels)
        history.append({"epoch": epoch, "train_loss": train_total / len(train),
                        "validation_loss": validation_total / len(validation),
                        "validation_metrics": {**threshold_metrics(truth, scores), **ranking_metrics(truth, scores)}})
    return model, history


def save_checkpoint(model: VariableLengthCNN, path: Path) -> None:
    """Persist model configuration and weights for inference replay."""
    payload = {"format_version": 1, "config": asdict(model.config), "state_dict": model.state_dict()}
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            torch.save(payload, file)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, device: str = "cpu") -> VariableLengthCNN:
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported CNN checkpoint version.")
    model = VariableLengthCNN(CNNConfig(**checkpoint["config"])).to(resolved)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
