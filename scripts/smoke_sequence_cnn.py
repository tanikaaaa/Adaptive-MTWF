"""Smoke-test the generic CNN with generated numerical sequences only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from adaptive_mtwf.cnn import CNNConfig
from adaptive_mtwf.data.batching import SequenceItem, collate_sequences
from adaptive_mtwf.sequence_training import TrainingConfig, load_checkpoint, resolve_device, save_checkpoint, train_sequences


def artificial_sequences(seed: int, count: int) -> list[SequenceItem]:
    """Generate two generic float channels and four binary latent-factor labels."""
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(count):
        length = int(rng.integers(17, 81))
        factors = rng.normal(size=2)
        channels = rng.normal(scale=0.4, size=(2, length)) + factors[:, None]
        a, b = factors
        target = np.asarray([a > 0, b > 0, a + b > 0, a - b > 0], dtype=np.float32)
        samples.append(SequenceItem(f"artificial-{seed}-{index}", channels.astype(np.float32), target))
    return samples


def run_smoke(device: str, epochs: int, optimizer: str, checkpoint: Path) -> dict:
    resolved = resolve_device(device)
    config = TrainingConfig(epochs=epochs, optimizer=optimizer, device=device)
    train, validation = artificial_sequences(9, 12), artificial_sequences(10, 6)
    start = time.perf_counter()
    model, history = train_sequences(CNNConfig(num_classes=4), train, validation, config)
    save_checkpoint(model, checkpoint)
    loaded = load_checkpoint(checkpoint, device=device)
    batch = collate_sequences(validation)
    inputs, mask = (torch.from_numpy(array).to(resolved) for array in (batch.inputs, batch.valid_mask))
    with torch.inference_mode():
        first, restored = model(inputs, mask), loaded(inputs, mask)
        torch.testing.assert_close(first, restored, rtol=0, atol=0)
    return {"device": device, "optimizer": optimizer, "parameters": sum(p.numel() for p in model.parameters()),
            "validation_input_shape": list(inputs.shape), "logit_shape": list(restored.shape),
            "checkpoint_replay_exact": True, "runtime_seconds": time.perf_counter() - start, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        with TemporaryDirectory() as directory:
            report = run_smoke(args.device, args.epochs, args.optimizer, Path(directory) / "artificial.pt")
    else:
        report = run_smoke(args.device, args.epochs, args.optimizer, args.checkpoint)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
