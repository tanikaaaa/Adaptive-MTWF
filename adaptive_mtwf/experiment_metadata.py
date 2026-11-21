"""Domain-independent split checks, result serialization, and fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .data.recipes import json_bytes
from .metrics import ThresholdCalibration, calibrate_threshold


def derive_seed(base_seed: int, namespace: str) -> int:
    """Derive an independent unsigned 63-bit stream seed from a named namespace."""
    if type(base_seed) is not int or base_seed < 0 or not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("Require a nonnegative integer seed and a nonblank namespace.")
    digest = hashlib.sha256(json_bytes(["experiment-seed-v1", base_seed, namespace])).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def positive_weights(*, split: str, positive_counts: ArrayLike, sample_count: int) -> np.ndarray:
    """Return (N - positives) / positives from explicit development-train counts."""
    if split != "development_train":
        raise ValueError("Positive weights require development-train counts.")
    counts = np.asarray(positive_counts)
    if type(sample_count) is not int or sample_count < 1 or counts.ndim != 1 or not counts.size or counts.dtype.kind not in "iu":
        raise ValueError("Require a positive sample count and a vector of integer counts.")
    if np.any((counts < 0) | (counts > sample_count)):
        raise ValueError("Positive counts must be in the sample-count range.")
    if np.any(counts == 0):
        raise ValueError("Positive weights are undefined for a class with zero positives.")
    return (sample_count - counts.astype(float)) / counts


def fit_validation_threshold(*, split: str, targets: ArrayLike, probabilities: ArrayLike) -> ThresholdCalibration:
    """Calibrate on the declared validation split only."""
    if split != "validation":
        raise ValueError("Threshold calibration requires the validation split.")
    return calibrate_threshold(targets, probabilities)


def reproducibility_fingerprint(config: dict, *, source_manifest: bytes, validation_manifest: bytes,
                                code_revision: str, seed: int) -> dict:
    """Digest canonical configuration and exact caller-supplied manifest bytes."""
    fields = {"config_sha256": hashlib.sha256(json_bytes(config)).hexdigest(),
              "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
              "validation_manifest_sha256": hashlib.sha256(validation_manifest).hexdigest(),
              "code_revision": code_revision, "seed": seed}
    return {**fields, "fingerprint_sha256": hashlib.sha256(json_bytes(fields)).hexdigest()}


@dataclass(frozen=True)
class ExperimentResult:
    """JSON result schema independent of the model and data provider."""

    config: dict
    git_commit: str
    timestamp_utc: str
    seeds: dict[str, int]
    device: str
    pytorch_version: str
    parameter_count: int
    train_size: int
    validation_size: int
    epoch_history: list[dict]
    best_epoch: Optional[int]
    best_validation_loss: Optional[float]
    calibrated_threshold: Optional[float]
    final_metrics: dict
    truncation_statistics: dict
    runtime_seconds: float

    def __post_init__(self) -> None:
        # Capture a JSON copy so later mutations of the caller's configuration
        # cannot change the recorded snapshot.
        object.__setattr__(self, "config", json.loads(json_bytes(self.config)))

    def to_bytes(self) -> bytes:
        return json_bytes(asdict(self))

    @classmethod
    def from_bytes(cls, raw: bytes) -> ExperimentResult:
        return cls(**json.loads(raw))
