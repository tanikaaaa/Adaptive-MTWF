"""Training, evaluation and epoch-boundary resume for numerical sequence batches."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, Subset

from .cnn import make_bce_loss
from .data.batching import SequenceBatch, SequenceItem, collate_sequences
from .data.recipes import json_bytes
from .experiment_metadata import ExperimentResult, derive_seed, fit_validation_threshold, reproducibility_fingerprint
from .metrics import ranking_metrics, threshold_metrics
from .sequence_training import TrainingConfig, resolve_device

Metric = Callable[[np.ndarray, np.ndarray, float], dict]
_RNG_LOCK = RLock()
_LOCK_FILENAME = ".experiment.lock"


def default_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    return {**threshold_metrics(targets, probabilities, threshold=threshold), **ranking_metrics(targets, probabilities)}


@dataclass(frozen=True)
class DataSource:
    """A caller-identified source of SequenceItems or loader batches (x, mask, y).

    Loaders may also yield SequenceBatch to report original lengths. IDs and
    content hashes are caller declarations. If the loader/dataset exposes a
    split, it must agree with the declared split.
    """

    split: str
    identifier: str
    data: Sequence[SequenceItem] | Dataset | DataLoader
    content_hash: Optional[str] = None

    def validate(self, expected: str) -> None:
        if self.split != expected:
            raise ValueError(f"Expected the {expected} split, got {self.split!r}.")
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("A caller-supplied dataset identifier is required.")
        pending = [self.data]
        while pending:
            obj = pending.pop()
            if getattr(obj, "split", expected) != expected:
                raise ValueError(f"Data or loader split does not match {expected}.")
            if isinstance(obj, (DataLoader, Subset)):
                pending.append(obj.dataset)
            if isinstance(obj, ConcatDataset):
                pending.extend(obj.datasets)
        if isinstance(self.data, DataLoader):
            if self.data.persistent_workers:
                raise ValueError("Persistent loader workers are unsupported for deterministic epoch replay.")
        elif (not isinstance(self.data, (Sequence, Dataset)) or isinstance(self.data, IterableDataset)
              or not len(self.data)):
            raise ValueError("Supply a nonempty indexable dataset of SequenceItems or a DataLoader.")

    def provenance(self) -> dict:
        info = {"split": self.split, "identifier": self.identifier, "content_hash": self.content_hash}
        if isinstance(self.data, DataLoader):
            info["loader"] = {"batch_size": self.data.batch_size, "drop_last": self.data.drop_last,
                              "num_workers": self.data.num_workers, "sampler": type(self.data.sampler).__qualname__,
                              "batch_sampler": type(self.data.batch_sampler).__qualname__}
        return info


@dataclass(frozen=True)
class Monitor:
    metric: str = "validation_loss"
    mode: str = "min"

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max") or not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("Specify a metric and a min or max mode.")

    def improves(self, value: float, reference: Optional[float], delta: float = 0.0) -> bool:
        if reference is None:
            return True
        return value < reference - delta if self.mode == "min" else value > reference + delta


@dataclass(frozen=True)
class EarlyStopping:
    monitor: Monitor = Monitor()
    patience: int = 5
    min_delta: float = 0.0001

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early stopping requires positive patience and a finite nonnegative delta.")

    def stopped(self, history: Sequence[dict]) -> bool:
        best, stale = None, 0
        for row in history:
            value = _monitor_value(row, self.monitor)
            if self.monitor.improves(value, best, self.min_delta):
                best, stale = value, 0
            else:
                stale += 1
            if stale >= self.patience:
                return True
        return False


@dataclass(frozen=True)
class RunnerConfig:
    num_classes: int
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(device="auto"))
    weight_decay: float = 0.0001
    gradient_clip: Optional[float] = 1.0
    pos_weight: Optional[tuple[float, ...]] = None
    checkpoint: Monitor = Monitor()
    early_stopping: Optional[EarlyStopping] = None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.num_classes) is not int or self.num_classes < 1 or type(self.version) is not int or self.version != 1:
            raise ValueError("Require a positive class count and runner config version 1.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("Gradient clipping must be finite and positive when configured.")
        if self.pos_weight is not None:
            weights = tuple(float(value) for value in self.pos_weight)
            if len(weights) != self.num_classes or any(not math.isfinite(value) or value < 0 for value in weights):
                raise ValueError("pos_weight requires one finite nonnegative value per class.")
            object.__setattr__(self, "pos_weight", weights)


@dataclass(frozen=True)
class RunMetadata:
    experiment_name: str
    code_revision: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("Experiment name must be nonblank.")


def select_device(name: str) -> torch.device:
    return resolve_device(("mps" if torch.backends.mps.is_available() else "cpu") if name == "auto" else name)


def _seed_all(base_seed: int, namespace: str) -> dict[str, int]:
    seeds = {name: derive_seed(base_seed, f"{namespace}:{name}") for name in ("python", "numpy", "pytorch")}
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"] % 2**32)
    seeds["numpy"] %= 2**32
    torch.manual_seed(seeds["pytorch"])
    return seeds


def _batches(source: DataSource, config: RunnerConfig, epoch: int):
    data, settings = source.data, config.training
    seed = derive_seed(settings.seed, f"loader:{source.split}:{epoch if source.split == 'development_train' else 0}")
    if isinstance(data, DataLoader):
        samplers = (data, data.sampler, getattr(data.batch_sampler, "sampler", None))
        generators = {id(gen): gen for obj in samplers if isinstance(gen := getattr(obj, "generator", None), torch.Generator)}
        for index, generator in enumerate(generators.values()):
            generator.manual_seed(derive_seed(seed, str(index)))
        yield from data
        return
    order = np.random.default_rng(seed).permutation(len(data)) if source.split == "development_train" else range(len(data))
    for start in range(0, len(order), settings.batch_size):
        items = [data[index] for index in order[start:start + settings.batch_size]]
        if not all(isinstance(item, SequenceItem) for item in items):
            raise ValueError("Indexable sources must yield SequenceItems.")
        yield collate_sequences(items, max_length=settings.max_length)


def _unpack(batch, config: RunnerConfig, device: torch.device):
    if isinstance(batch, SequenceBatch):
        arrays = batch.inputs, batch.valid_mask, batch.targets
        original = batch.original_lengths.tolist()
        ids = batch.sample_ids
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        arrays, original, ids = batch, None, None
    else:
        raise ValueError("A batch must be a SequenceBatch or (inputs, mask, targets).")
    inputs, mask, targets = (torch.as_tensor(array, device=device) for array in arrays)
    if inputs.ndim != 3 or min(inputs.shape) < 1 or not inputs.is_floating_point():
        raise ValueError("Inputs must be floating point with nonempty (batch, channels, length) shape.")
    if inputs.shape[2] > config.training.max_length:
        raise ValueError("A prebatched sequence exceeds the configured safety ceiling.")
    if inputs.shape[0] > config.training.batch_size:
        raise ValueError("A loader batch exceeds the configured batch size.")
    if not inputs.isfinite().all().item():
        raise FloatingPointError("Nonfinite input channels.")
    if (mask.shape != (inputs.shape[0], inputs.shape[2]) or mask.dtype != torch.bool
            or not mask[:, 0].all().item() or (mask[:, 1:] & ~mask[:, :-1]).any().item()):
        raise ValueError("Masks must be boolean and mark a nonempty contiguous prefix per row.")
    if targets.shape != (inputs.shape[0], config.num_classes) or targets.is_complex() or not ((targets == 0) | (targets == 1)).all().item():
        raise ValueError("Targets must be binary with shape (batch, configured classes).")
    retained = mask.sum(dim=1).tolist()
    if original is not None and (len(original) != inputs.shape[0] or len(ids) != inputs.shape[0]
                                 or any(length < kept for length, kept in zip(original, retained))
                                 or batch.retained_lengths.tolist() != retained):
        raise ValueError("SequenceBatch lengths or sample IDs disagree with the batch mask.")
    return inputs, mask, targets.float(), original, retained


def _logits(model: nn.Module, inputs, mask, targets):
    logits = model(inputs, mask)
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or logits.shape != targets.shape:
        raise ValueError("Model logits must be floating point and match the target dimensions.")
    if not logits.isfinite().all().item():
        raise FloatingPointError("Nonfinite model logits.")
    return logits


def _pass(model, source, config, device, criterion, epoch, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_total, count, batches = 0.0, 0, 0
    original_lengths, retained_lengths, probabilities, labels = [], [], [], []
    start = time.perf_counter()
    with torch.set_grad_enabled(train):
        for batch in _batches(source, config, epoch):
            inputs, mask, targets, original, retained = _unpack(batch, config, device)
            logits = _logits(model, inputs, mask, targets)
            loss = criterion(logits, targets)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite loss.")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not grads or not torch.stack([grad.isfinite().all() for grad in grads]).all().item():
                    raise FloatingPointError("Missing or nonfinite gradients.")
                if config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                probabilities.append(logits.sigmoid().cpu().numpy())
                labels.append(targets.cpu().numpy())
            size = targets.shape[0]
            loss_total += value * size
            count += size
            batches += 1
            if original is not None:
                original_lengths.extend(original)
                retained_lengths.extend(retained)
    if not count:
        raise ValueError(f"{source.split} produced no samples.")
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    truncation = {"sample_count": count, "samples_with_original_lengths": len(original_lengths)}
    if original_lengths:
        fractions = np.asarray(retained_lengths) / np.asarray(original_lengths)
        truncation.update(truncated=int(np.count_nonzero(fractions < 1)), mean_retained_fraction=float(fractions.mean()),
                          median_retained_fraction=float(np.median(fractions)), max_original_length=max(original_lengths))
    return ({"loss": loss_total / count, "samples": count, "batches": batches, "runtime_seconds": elapsed,
             "samples_per_second": count / elapsed, "truncation": truncation},
            None if train else np.concatenate(labels), None if train else np.concatenate(probabilities))


def _monitor_value(row: dict, monitor: Monitor) -> float:
    value = row["validation_loss"] if monitor.metric == "validation_loss" else row["validation_metrics"].get(monitor.metric)
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"Checkpoint or early stopping metric {monitor.metric!r} must be a defined finite validation scalar.")
    return float(value)


def _atomic_write(path: Path, data) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            if isinstance(data, bytes):
                file.write(data)
            else:
                torch.save(data, file)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def restore_checkpoint(model: nn.Module, path: Path, *, optimizer=None) -> dict:
    """Restore the supplied model, optionally including optimizer buffers."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported experiment checkpoint format.")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _revision(metadata: RunMetadata) -> str:
    if metadata.code_revision is not None:
        return metadata.code_revision
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _history_files(output: Path, history: list[dict]) -> None:
    _atomic_write(output / "history.json", json_bytes(history))
    flat = [{**{key: value for key, value in row.items() if type(value) in (int, float)},
             **{f"validation_{key}": value for key, value in row["validation_metrics"].items()
                if value is None or type(value) in (float, int)}} for row in history]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in flat for key in row)))
    writer.writeheader()
    writer.writerows(flat)
    _atomic_write(output / "history.csv", stream.getvalue().encode("utf-8"))


def _snapshot(model, config, metadata, train, validation, metrics, device, revision) -> dict:
    architecture = getattr(model, "config", None)
    snapshot = {
        "experiment_name": metadata.experiment_name, "settings": asdict(config), "optimizer": config.training.optimizer,
        "loss": "BCEWithLogitsLoss", "selected_device": str(device), "code_revision": revision,
        "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "metric_function": f"{getattr(metrics, '__module__', '')}.{getattr(metrics, '__qualname__', type(metrics).__qualname__)}",
        "model_config": asdict(architecture) if is_dataclass(architecture) else None,
        "model_state_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                               for key, value in model.state_dict().items()},
        "train": train.provenance(), "validation": validation.provenance(), "extra": metadata.extra,
    }
    return json.loads(json_bytes(snapshot))


def _resume_key(snapshot):
    snapshot = json.loads(json_bytes(snapshot))
    del snapshot["settings"]["training"]["epochs"]
    snapshot.pop("dimensions", None)  # Checked against the current first batches separately.
    snapshot.pop("initial_model_sha256", None)  # The checkpoint supplies the original initialization hash.
    return hashlib.sha256(json_bytes(snapshot)).hexdigest()


def _checked_metrics(metrics, truth, scores, threshold):
    metrics_out = metrics(truth, scores, threshold)
    if not isinstance(metrics_out, dict) or any(not isinstance(key, str) for key in metrics_out):
        raise ValueError("Metric function must return a dictionary with string keys.")
    return json.loads(json_bytes(metrics_out))


def _preflight(model, sources, config, device, metrics):
    """Validate first batch interfaces before allocating an optimizer or writing files."""
    model.eval()
    dimensions = []
    for source in sources:
        batches = _batches(source, config, 0)
        try:
            batch = next(batches, None)
            if batch is None:
                raise ValueError(f"{source.split} produced no samples.")
            inputs, mask, targets, _, _ = _unpack(batch, config, device)
            with torch.no_grad():
                logits = _logits(model, inputs, mask, targets)
            dimensions.append(inputs.shape[1])
            if source.split == "validation":
                fields = _checked_metrics(metrics, targets.cpu().numpy(), logits.sigmoid().cpu().numpy(), 0.5).keys()
                monitors = [config.checkpoint] + ([config.early_stopping.monitor] if config.early_stopping else [])
                for monitor in monitors:
                    if monitor.metric != "validation_loss" and monitor.metric not in fields:
                        raise ValueError(f"Unknown validation metric: {monitor.metric!r}.")
        finally:
            batches.close()
    if dimensions[0] != dimensions[1]:
        raise ValueError("Training and validation input channel counts differ.")
    return {"input_channels": dimensions[0], "target_classes": config.num_classes, "variable_length": True}


def _finalize(output, model, validation, config, snapshot, revision, device, criterion,
              metrics, history, initialization_seeds, elapsed):
    best = restore_checkpoint(model, output / "best.pt")
    _seed_all(config.training.seed, "validation")
    val, truth, scores = _pass(model, validation, config, device, criterion, 0)
    calibration = fit_validation_threshold(split=validation.split, targets=truth, probabilities=scores)
    final = {"validation_loss": val["loss"], "fixed": _checked_metrics(metrics, truth, scores, 0.5),
             "calibrated": _checked_metrics(metrics, truth, scores, calibration.threshold)}
    result = ExperimentResult(config=snapshot, git_commit=revision, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                              seeds={"base": config.training.seed, **initialization_seeds}, device=str(device),
                              pytorch_version=str(torch.__version__), parameter_count=best["parameter_count"],
                              train_size=history[-1]["train_samples"], validation_size=val["samples"], epoch_history=history,
                              best_epoch=best["epoch"], best_validation_loss=best["validation_loss"],
                              calibrated_threshold=calibration.threshold, final_metrics=final,
                              truncation_statistics={"train_by_epoch": [row["train_truncation"] for row in history],
                                                     "validation": val["truncation"]},
                              runtime_seconds=elapsed + val["runtime_seconds"])
    artifacts = {
        "best_checkpoint.json": {key: best[key] for key in ("epoch", "validation_metric", "validation_loss", "parameter_count")},
        "threshold.json": {"split": validation.split, "fixed_threshold": 0.5, **asdict(calibration)},
        "final_validation.json": final,
        "runtime.json": {"total_seconds": result.runtime_seconds,
                         "training_seconds": sum(row["train_runtime_seconds"] for row in history),
                         "epoch_validation_seconds": sum(row["validation_runtime_seconds"] for row in history),
                         "final_validation_seconds": val["runtime_seconds"]},
    }
    for name, record in artifacts.items():
        _atomic_write(output / name, json_bytes(record))
    _atomic_write(output / "metadata.json", result.to_bytes())
    _atomic_write(output / "status.json", json_bytes({"status": "complete", "last_epoch": history[-1]["epoch"]}))
    return result


def run_experiment(*, train: DataSource, validation: DataSource, model: nn.Module, config: RunnerConfig,
                   output: Path, metadata: RunMetadata, metrics: Metric = default_metrics,
                   resume: bool = False) -> ExperimentResult:
    """Run an experiment on precomputed numerical data and reload its best weights.

    The supplied model is used as initialized. To seed its initialization,
    construct it with a seeded factory (as run_seeds does). Epoch-boundary
    replay reseeds RNGs and standard loader generators. Custom stateful data
    providers must manage their own state. Concurrent writes are rejected.
    An increased epoch limit is the only settings change permitted on resume.
    """
    train.validate("development_train")
    validation.validate("validation")
    train_dataset = train.data.dataset if isinstance(train.data, DataLoader) else train.data
    validation_dataset = validation.data.dataset if isinstance(validation.data, DataLoader) else validation.data
    if train.identifier == validation.identifier or train_dataset is validation_dataset or (
        train.content_hash is not None and train.content_hash == validation.content_hash
    ):
        raise ValueError("Training and validation must be distinct sources.")
    if output.is_symlink() or output.exists() and not output.is_dir():
        raise ValueError("Output must be a normal directory.")
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output / _LOCK_FILENAME), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"Another writer is using {output}.") from error
    try:
        with _RNG_LOCK:
            return _run_experiment(train, validation, model, config, output, metadata, metrics, resume)
    finally:
        lock.release()


def _run_experiment(train, validation, model, config, output, metadata, metrics, resume):
    started = time.perf_counter()
    latest = output / "latest.pt"
    if resume and not latest.is_file():
        raise ValueError("Resume requires an existing latest.pt checkpoint.")
    if not resume and any(path.name != _LOCK_FILENAME for path in output.iterdir()):
        raise ValueError("A new experiment requires an empty output directory.")

    device = select_device(config.training.device)
    revision = _revision(metadata)
    model.to(device)
    model.zero_grad(set_to_none=True)
    snapshot = _snapshot(model, config, metadata, train, validation, metrics, device, revision)
    state = torch.load(latest, map_location="cpu", weights_only=True) if resume else None
    if state is not None and (state.get("format_version") != 1 or state.get("resume_key") != _resume_key(snapshot)
                              or config.training.epochs < state["config"]["settings"]["training"]["epochs"]):
        raise ValueError("Incompatible resume configuration or provenance.")
    initialization_seeds = _seed_all(config.training.seed, "initialization")
    if state is not None:
        model.load_state_dict(state["model_state"])
        snapshot["initial_model_sha256"] = state["config"]["initial_model_sha256"]
    else:
        buffer = io.BytesIO()
        torch.save(_cpu_copy(model.state_dict()), buffer)
        snapshot["initial_model_sha256"] = hashlib.sha256(buffer.getbuffer()).hexdigest()
    snapshot["dimensions"] = _preflight(model, (train, validation), config, device, metrics)
    if state is not None and state["config"]["dimensions"] != snapshot["dimensions"]:
        raise ValueError("Incompatible resume input dimensions.")
    criterion = make_bce_loss(config.num_classes, pos_weight=config.pos_weight, device=device)
    optimizer_type = torch.optim.Adam if config.training.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=config.training.learning_rate, weight_decay=config.weight_decay)
    if state is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    history = state["history"] if state is not None else []
    best = state["best"] if state is not None else None
    previous_runtime = state["runtime_seconds"] if state is not None else 0.0
    last_epoch = state["epoch"] if state is not None else 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    fingerprint = reproducibility_fingerprint(snapshot, source_manifest=json_bytes(train.provenance()),
                                              validation_manifest=json_bytes(validation.provenance()),
                                              code_revision=revision, seed=config.training.seed)
    _atomic_write(output / "status.json", json_bytes({"status": "running"}))
    _atomic_write(output / "config.json", json_bytes(snapshot))
    _atomic_write(output / "fingerprint.json", json_bytes(fingerprint))

    for epoch in range(last_epoch + 1, config.training.epochs + 1):
        if config.early_stopping and config.early_stopping.stopped(history):
            break
        epoch_start = time.perf_counter()
        _seed_all(config.training.seed, f"train:{epoch}")
        trained, _, _ = _pass(model, train, config, device, criterion, epoch, optimizer)
        _seed_all(config.training.seed, "validation")
        val, truth, scores = _pass(model, validation, config, device, criterion, epoch)
        row = {"epoch": epoch, "train_loss": trained["loss"], "validation_loss": val["loss"],
               "validation_metrics": _checked_metrics(metrics, truth, scores, 0.5), "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_runtime_seconds": time.perf_counter() - epoch_start,
               **{f"{split}_{key}": data[key] for split, data in (("train", trained), ("validation", val))
                  for key in ("runtime_seconds", "samples_per_second", "samples", "batches", "truncation")}}
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            row["mps_allocated_snapshot_bytes"] = torch.mps.current_allocated_memory()
        score = _monitor_value(row, config.checkpoint)
        if config.early_stopping is not None:
            _monitor_value(row, config.early_stopping.monitor)
        history.append(row)
        checkpoint = {"format_version": 1, "model_state": _cpu_copy(model.state_dict()),
                      "optimizer_state": _cpu_copy(optimizer.state_dict()), "epoch": epoch,
                      "validation_metric": {"name": config.checkpoint.metric, "value": score, "mode": config.checkpoint.mode},
                      "validation_loss": val["loss"], "config": snapshot, "seed": config.training.seed,
                      "parameter_count": parameter_count}
        improved = config.checkpoint.improves(score, None if best is None else best["validation_metric"]["value"])
        if improved:
            best = checkpoint
        # Latest commits the epoch and embeds the best snapshot. A restart can
        # recover both even if writing best.pt or history was interrupted.
        _atomic_write(latest, {**checkpoint, "best": best, "history": history, "resume_key": _resume_key(snapshot),
                               "runtime_seconds": previous_runtime + time.perf_counter() - started})
        if improved:
            _atomic_write(output / "best.pt", best)
        _history_files(output, history)

    _atomic_write(output / "best.pt", best)
    _history_files(output, history)
    return _finalize(output, model, validation, config, snapshot, revision, device, criterion,
                     metrics, history, initialization_seeds, previous_runtime + time.perf_counter() - started)


def aggregate_results(results: Sequence[ExperimentResult]) -> dict:
    """Summarize finite scalar metrics across completed runs, using population SD."""
    values: dict[str, list[float]] = {}
    for result in results:
        for group, nested in result.final_metrics.items():
            for key, value in (nested.items() if isinstance(nested, dict) else [(None, nested)]):
                if type(value) in (int, float) and math.isfinite(value):
                    values.setdefault(f"{group}.{key}" if key is not None else group, []).append(float(value))
    return {"successful_runs": len(results), "std_convention": "population",
            "metrics": {key: {"mean": float(np.mean(series)), "std": float(np.std(series)), "min": min(series),
                              "max": max(series), "successful_runs": len(series)} for key, series in values.items()}}


def run_seeds(*, seeds: Sequence[int], model_factory: Callable[[], nn.Module], train: DataSource, validation: DataSource,
              config: RunnerConfig, root: Path, metadata: RunMetadata, metrics: Metric = default_metrics) -> dict:
    """Run independent jobs in seed-specific directories, failing on the first error."""
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Supply a nonempty list of distinct valid seeds.")
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise ValueError("The multi-seed output root must be a new or empty directory.")
    root.mkdir(parents=True, exist_ok=True)
    shared = asdict(config)
    del shared["training"]["seed"]
    plan = {"seeds": list(seeds), "shared_config_sha256": hashlib.sha256(json_bytes(shared)).hexdigest()}
    with FileLock(str(root / _LOCK_FILENAME), timeout=0):
        if any(path.name != _LOCK_FILENAME for path in root.iterdir()):
            raise ValueError("The multi-seed output root must be a new or empty directory.")
        _atomic_write(root / "seed_plan.json", json_bytes(plan))
    runs = []
    for seed in seeds:
        with _RNG_LOCK:
            _seed_all(seed, "initialization")
            result = run_experiment(train=train, validation=validation, model=model_factory(),
                                    config=replace(config, training=replace(config.training, seed=seed)),
                                    output=root / f"seed_{seed}", metadata=metadata, metrics=metrics)
        runs.append(result)
        summary = {**plan, "completed_seeds": list(seeds[:len(runs)]), **aggregate_results(runs)}
        _atomic_write(root / "aggregate.json", json_bytes(summary))
    return summary
