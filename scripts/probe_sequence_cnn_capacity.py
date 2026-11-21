"""Measure one generic CNN optimization step at each artificial batch size."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from adaptive_mtwf.cnn import CNNConfig, VariableLengthCNN, make_bce_loss
from adaptive_mtwf.experiment_metadata import derive_seed
from adaptive_mtwf.sequence_training import resolve_device

CONFIG = Path(__file__).resolve().parents[1] / "configs/artificial_cnn_capacity_v1.json"
FROZEN = {"name": "artificial_cnn_capacity", "version": 1, "input_source": "artificial", "device": "mps",
          "architecture": "VariableLengthCNN default backbone", "channels": 2, "classes": 4,
          "sequence_length": 30000, "candidate_batch_sizes": [4, 8, 16], "optimizer": "adamw",
          "optimizer_steps_per_candidate": 1, "loss": "BCEWithLogitsLoss", "dataset_access": False}


def validate_config(config: dict) -> None:
    """Validate the frozen artificial-only capacity probe configuration."""
    if config.keys() != (FROZEN.keys() | {"seed"}):
        raise ValueError("Unexpected or missing capacity configuration fields.")
    for field, value in FROZEN.items():
        if type(config[field]) is not type(value) or config[field] != value:
            raise ValueError(f"Invalid frozen capacity setting: {field}.")
    derive_seed(config["seed"], "capacity-validation")


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _snapshot(device: torch.device) -> dict:
    if device.type != "mps":
        return {}
    return {name: getattr(torch.mps, function)() for name, function in (
        ("allocated_bytes", "current_allocated_memory"), ("driver_allocated_bytes", "driver_allocated_memory")
    ) if hasattr(torch.mps, function)}


def probe_step(batch_size: int, length: int, *, device: torch.device, seed: int) -> dict:
    """Allocate generated floats and binary labels, then run one optimizer step."""
    torch.manual_seed(seed)
    _synchronize(device)
    before = _snapshot(device)
    start = time.perf_counter()
    model = VariableLengthCNN(CNNConfig(4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.randn(batch_size, 2, length, device=device)
    mask = torch.ones(batch_size, length, device=device, dtype=torch.bool)
    targets = (torch.rand(batch_size, 4, device=device) >= 0.5).to(dtype=inputs.dtype)
    logits = model(inputs, mask)
    loss = make_bce_loss(4)(logits, targets)
    finite_loss = math.isfinite(loss.item())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    finite_gradients = all(grad is not None for grad in gradients) and bool(
        torch.stack([grad.isfinite().all() for grad in gradients]).all().item())
    after_backward = _snapshot(device)
    if finite_loss and finite_gradients:
        optimizer.step()
    _synchronize(device)
    after_step = _snapshot(device)
    return {"batch_size": batch_size, "input_shape": list(inputs.shape), "output_shape": list(logits.shape),
            "status": "passed" if finite_loss and finite_gradients else "failed",
            "finite_loss": finite_loss, "finite_gradients": finite_gradients,
            "loss": loss.item() if finite_loss else None, "runtime_seconds": time.perf_counter() - start,
            "memory_snapshots": {"before": before, "after_backward": after_backward, "after_step": after_step}}


def run_probe(config: dict) -> dict:
    validate_config(config)
    sizes = config["candidate_batch_sizes"]
    try:
        device = resolve_device(config["device"])
    except RuntimeError as error:
        return {"config": config, "pytorch_version": str(torch.__version__), "batch_size_recommendation": None,
                "results": [{"batch_size": size, "status": "skipped", "reason": str(error)} for size in sizes]}

    results = []
    for size in sizes:
        torch.mps.empty_cache()
        try:
            result = probe_step(size, config["sequence_length"], device=device,
                                seed=derive_seed(config["seed"], f"capacity-{size}"))
        except RuntimeError as error:
            result = {"batch_size": size, "status": "failed", "reason": str(error)}
        results.append(result)
        if result["status"] != "passed":
            results.extend({"batch_size": remaining, "status": "skipped", "reason": "Earlier candidate failed."}
                           for remaining in sizes[len(results):])
            break
    passed = [row["batch_size"] for row in results if row["status"] == "passed"]
    return {"config": config, "pytorch_version": str(torch.__version__), "results": results,
            "memory_policy": "Snapshots are not peaks. Use one candidate below the largest passing size.",
            "batch_size_recommendation": passed[-2] if len(passed) > 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(json.loads(args.config.read_bytes()))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
