"""Normalize source times and prepare fixed-length model inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import MAX_CELLS, MAX_SOURCE_CELLS, CanonicalTrace, WebsiteLabel
from .validate import TraceValidationError, ValidationIssue, ValidationResult, validate_single_tab_trace, validate_trace


def _subtract_time_origin(values: NDArray[np.generic], origin: np.generic) -> NDArray[np.float64]:
    """Rebase times at their minimum before converting to the output dtype."""
    if values.dtype.kind in "iu":
        # Unsigned subtraction gives the exact nonnegative span of signed or
        # unsigned 64-bit timestamps before any rounding to float64.
        return (values.astype(np.uint64) - origin.astype(np.uint64)).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        time_dtype = np.result_type(values.dtype, np.float64)
        return np.subtract(values, origin, dtype=time_dtype).astype(np.float64)


def make_trace(
    timestamps: ArrayLike,
    directions: ArrayLike,
    *,
    labels: tuple[WebsiteLabel, ...],
    trace_id: str,
    source_dataset: str,
    metadata: Optional[dict[str, Any]] = None,
) -> CanonicalTrace:
    """Validate observed cells, then subtract the first timestamp.

    The returned trace owns its cell arrays and retains all observations in
    their original order. Times are float64 in the original units, and
    directions are int8. Labels and IDs use the schema's types.
    """
    arrays = []
    for field, values in (("timestamps", timestamps), ("directions", directions)):
        try:
            arrays.append(np.asarray(values))
        except (TypeError, ValueError, OverflowError) as exc:
            issue = ValidationIssue("invalid_cells", field, f"{field} could not be read as a NumPy array.")
            raise TraceValidationError(ValidationResult((issue,))) from exc
    times, signed_directions = arrays
    trace = CanonicalTrace(
        timestamps=times,
        directions=signed_directions,
        labels=labels,
        trace_id=trace_id,
        source_dataset=source_dataset,
        valid_length=times.size,
        metadata={} if metadata is None else metadata,
    )
    validate_trace(trace, require_relative=False).raise_if_invalid()

    normalized = replace(
        trace,
        timestamps=_subtract_time_origin(times, times[0]),
        directions=signed_directions.astype(np.int8),
        metadata=dict(trace.metadata),
    )
    validate_trace(normalized).raise_if_invalid()
    return normalized


def cap_single_tab_trace(trace: CanonicalTrace) -> CanonicalTrace:
    """Keep up to 5,000 valid cells of a single tab as an unpadded trace.

    Eligibility is checked on the entire supplied trace, including cells after
    the cap. The selected cells retain their order and relative time origin.
    """
    validate_single_tab_trace(trace).raise_if_invalid()
    length = min(trace.valid_length, MAX_SOURCE_CELLS)
    return replace(
        trace,
        timestamps=trace.timestamps[:length].copy(),
        directions=trace.directions[:length].copy(),
        valid_length=length,
        metadata=dict(trace.metadata),
        mask=None,
    )


def canonicalize_trace(trace: CanonicalTrace) -> CanonicalTrace:
    """Validate the entire trace and produce the legacy 5,000-cell representation.

    Return separate float64 timestamps, int8 directions, and a boolean mask.
    All three arrays have shape ``(5000,)``, and both channels are zero in
    padding. Already fixed-length traces can be converted again.
    """
    validate_trace(trace).raise_if_invalid()
    valid_length = min(trace.valid_length, MAX_CELLS)
    timestamps = np.zeros(MAX_CELLS, dtype=np.float64)
    directions = np.zeros(MAX_CELLS, dtype=np.int8)
    mask = np.zeros(MAX_CELLS, dtype=np.bool_)
    timestamps[:valid_length] = trace.timestamps[:valid_length]
    directions[:valid_length] = trace.directions[:valid_length]
    mask[:valid_length] = True
    return replace(
        trace,
        timestamps=timestamps,
        directions=directions,
        valid_length=valid_length,
        metadata=dict(trace.metadata),
        mask=mask,
    )


def inter_arrival_times(trace: CanonicalTrace) -> NDArray[np.float64]:
    """Return consecutive valid timestamp differences, aligned with the cells.

    The first cell and padded positions receive 0.0. A zero delta is also a
    valid observation for equal consecutive timestamps. Use the trace's mask
    to identify padding when working with the fixed-length representation.
    """
    validate_trace(trace).raise_if_invalid()
    deltas = np.zeros(trace.timestamps.shape, dtype=np.float64)
    valid_times = trace.timestamps[: trace.valid_length].astype(np.float64, copy=False)
    deltas[1 : trace.valid_length] = np.diff(valid_times)
    return deltas
