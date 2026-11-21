"""NumPy collation for precomputed, variable-length numerical sequences."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_MAX_LENGTH = 30_000  # Computational ceiling; batches are padded dynamically.


def encode_multihot(labels: Iterable[Hashable], vocabulary: Sequence[Hashable]) -> NDArray[np.float32]:
    """Encode labels against an explicit ordered vocabulary of distinct classes."""
    positions = {label: index for index, label in enumerate(vocabulary)}
    if not positions or len(positions) != len(vocabulary):
        raise ValueError("The vocabulary must be nonempty and contain distinct labels.")
    target = np.zeros(len(vocabulary), dtype=np.float32)
    for label in labels:
        if label not in positions:
            raise ValueError(f"Label {label!r} is outside the supplied vocabulary.")
        target[positions[label]] = 1.0
    return target


@dataclass(frozen=True, eq=False)
class SequenceItem:
    sample_id: str
    channels: ArrayLike  # (channels, length)
    target: ArrayLike  # (classes,)


@dataclass(frozen=True, eq=False)
class SequenceBatch:
    sample_ids: tuple[str, ...]
    inputs: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    targets: NDArray[np.float32]
    original_lengths: NDArray[np.int64]
    retained_lengths: NDArray[np.int64]
    retained_fractions: NDArray[np.float64]
    truncated: NDArray[np.bool_]


def collate_sequences(items: Sequence[SequenceItem], *, max_length: int = DEFAULT_MAX_LENGTH) -> SequenceBatch:
    """Pad channels to the largest retained length and report every truncation.

    Each input must have the same number of channels, with at least one sample
    per sequence. Channels are converted to float32. Targets must be binary
    vectors of a shared width. The true prefix in ``valid_mask`` identifies
    retained cells regardless of their numeric values.
    """
    if not isinstance(max_length, Integral) or isinstance(max_length, (bool, np.bool_)) or max_length < 1:
        raise ValueError("The safety ceiling must be a positive integer.")
    if not items:
        raise ValueError("At least one sequence item is required.")

    arrays = []
    targets = []
    for item in items:
        if not isinstance(item.sample_id, str) or not item.sample_id.strip():
            raise ValueError("Each sequence needs a nonblank sample ID.")
        channels = np.asarray(item.channels)
        if channels.ndim != 2 or min(channels.shape) < 1 or channels.dtype.kind not in "iufb":
            raise ValueError(f"{item.sample_id}: expected a nonempty real (channels, length) array.")
        with np.errstate(over="ignore", invalid="ignore"):
            channels = channels.astype(np.float32, copy=False)
        if not np.isfinite(channels).all():
            raise ValueError(f"{item.sample_id}: channel values must be finite in float32.")
        target = np.asarray(item.target)
        if target.ndim != 1 or not target.size or target.dtype.kind not in "iufb" or not np.isin(target, (0, 1)).all():
            raise ValueError(f"{item.sample_id}: expected a nonempty binary target vector.")
        arrays.append(channels)
        targets.append(target)

    channel_count = arrays[0].shape[0]
    if any(channels.shape[0] != channel_count for channels in arrays):
        raise ValueError("All items must have the same channel count.")
    if any(target.shape != targets[0].shape for target in targets):
        raise ValueError("All items must have the same target width.")

    original_lengths = np.array([channels.shape[1] for channels in arrays], dtype=np.int64)
    retained_lengths = np.minimum(original_lengths, min(int(max_length), int(original_lengths.max())))
    batch_length = int(retained_lengths.max())
    inputs = np.zeros((len(items), channel_count, batch_length), dtype=np.float32)
    valid_mask = np.arange(batch_length)[None, :] < retained_lengths[:, None]
    for batch_row, (channels, length) in enumerate(zip(arrays, retained_lengths)):
        inputs[batch_row, :, :length] = channels[:, :length]
    return SequenceBatch(
        sample_ids=tuple(item.sample_id for item in items), inputs=inputs, valid_mask=valid_mask,
        targets=np.stack(targets).astype(np.float32, copy=False), original_lengths=original_lengths,
        retained_lengths=retained_lengths, retained_fractions=retained_lengths / original_lengths,
        truncated=retained_lengths < original_lengths,
    )
