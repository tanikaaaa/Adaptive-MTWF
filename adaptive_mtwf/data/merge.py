"""Build artificial multi-tab samples from explicit capture start offsets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from numbers import Integral
from typing import Literal, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonicalize import _subtract_time_origin, canonicalize_trace, make_trace
from .schema import MERGE_PROVENANCE_KEY, CanonicalTrace
from .validate import validate_single_tab_trace

TiePolicy = Literal["legacy_tmwf", "seeded_random"]


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("The RNG seed must be a nonnegative integer.")
    return int(seed)


def _prepare_start_offsets(
    start_offsets: ArrayLike, count: int
) -> tuple[list[Union[int, float]], NDArray[np.float64]]:
    offsets = np.asarray(start_offsets)
    if offsets.shape != (count,) or offsets.dtype.kind not in "iuf":
        raise ValueError("Provide one real start offset per trace.")
    if offsets.dtype.kind == "f":
        with np.errstate(over="ignore", invalid="ignore"):
            offsets = offsets.astype(np.float64, copy=False)
    if not np.isfinite(offsets).all():
        raise ValueError("Start offsets must be finite float64 values.")

    # Use the earliest start as the origin before addition to retain small
    # cell gaps when all supplied offsets share a large common origin.
    relative_offsets = _subtract_time_origin(offsets, offsets.min())
    if not np.isfinite(relative_offsets).all():
        raise ValueError("Relative start offsets must fit in float64.")
    return offsets.tolist(), relative_offsets


def _randomize_ties(
    order: NDArray[np.intp], timestamps: NDArray[np.float64], lengths: list[int], seed: int
) -> None:
    """Shuffle source slots within each exact tie and fill them in cell order."""
    sorted_times = timestamps[order]
    sources = np.repeat(np.arange(len(lengths)), lengths)[order]
    boundaries = np.flatnonzero(sorted_times[1:] != sorted_times[:-1]) + 1
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, order.size]
    for start, end in zip(starts, ends):
        # The stable sort made each source a contiguous part of the tie.
        if sources[start] == sources[end - 1]:
            continue
        source_slots = sources[start:end]
        shuffled_slots = rng.permutation(source_slots)
        group = order[start:end]
        original = group.copy()
        for source in np.unique(source_slots):
            group[shuffled_slots == source] = original[source_slots == source]


def merge_variable_length(
    traces: Sequence[CanonicalTrace],
    start_offsets: ArrayLike,
    *,
    tie_policy: TiePolicy = "seeded_random",
    seed: int = 0,
) -> CanonicalTrace:
    """Merge every supplied valid cell on an unpadded relative timeline.

    All sources must be single-tab traces in the same time units. Offsets can
    be signed. ``legacy_tmwf`` puts earlier sources first at exact ties.
    ``seeded_random`` shuffles cross-source slots with PCG64 at each exact
    float64 tie. Both preserve order within each source.
    """
    if len(traces) < 2:
        raise ValueError("At least two single-tab traces are required.")
    if tie_policy not in ("legacy_tmwf", "seeded_random"):
        raise ValueError(f"Unknown tie policy: {tie_policy!r}")
    seed = _validate_seed(seed)
    offsets, relative_offsets = _prepare_start_offsets(start_offsets, len(traces))
    for trace in traces:
        validate_single_tab_trace(trace).raise_if_invalid()

    with np.errstate(over="ignore", invalid="ignore"):
        shifted_times = np.concatenate([
            trace.timestamps[: trace.valid_length].astype(np.float64, copy=False) + offset
            for trace, offset in zip(traces, relative_offsets)
        ])
    if not np.isfinite(shifted_times).all():
        raise ValueError("Shifted timestamps must fit in float64.")

    directions = np.concatenate([trace.directions[: trace.valid_length] for trace in traces])
    order = np.argsort(shifted_times, kind="stable")
    if tie_policy == "seeded_random":
        _randomize_ties(order, shifted_times, [trace.valid_length for trace in traces], seed)
    constituents = [
        {
            "trace_id": trace.trace_id,
            "source_dataset": trace.source_dataset,
            "start_offset": offset,
            "valid_length": int(trace.valid_length),
        }
        for trace, offset in zip(traces, offsets)
    ]
    metadata = {MERGE_PROVENANCE_KEY: constituents, "tie_policy": tie_policy, "rng_seed": seed}
    identity = metadata if tie_policy == "seeded_random" else constituents
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return make_trace(
        shifted_times[order],
        directions[order],
        labels=tuple(dict.fromkeys(trace.labels[0] for trace in traces)),
        trace_id=f"merged-{digest}",
        source_dataset="synthetic",
        metadata=metadata,
    )


def merge_with_offsets(traces: Sequence[CanonicalTrace], start_offsets: ArrayLike) -> CanonicalTrace:
    """Return the legacy fixed 5,000-cell merge with source-priority ties."""
    merged = merge_variable_length(traces, start_offsets, tie_policy="legacy_tmwf")
    return canonicalize_trace(replace(merged, metadata={MERGE_PROVENANCE_KEY: merged.metadata[MERGE_PROVENANCE_KEY]}))
