"""Validate trace invariants and report data issues without editing traces."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .schema import MAX_CELLS, MERGE_PROVENANCE_KEY, CanonicalTrace


@dataclass(frozen=True)
class ValidationIssue:
    """Provide a stable issue code and field for dataset-level aggregation."""

    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Collect the issues found in one trace."""

    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        """Raise with the full validation result when any invariant fails."""
        if self.issues:
            raise TraceValidationError(self)


class TraceValidationError(ValueError):
    """An operation rejected a trace and retained its structured issues."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__("; ".join(issue.message for issue in result.issues))


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _is_label(value: object) -> bool:
    return _is_integer(value) or (isinstance(value, str) and bool(value.strip()))


def validate_trace(trace: CanonicalTrace, *, require_relative: bool = True) -> ValidationResult:
    """Check a canonical trace, allowing absolute timestamps during ingestion.

    With ``require_relative=False``, the original timestamp origin is accepted.
    Every cell in an unmasked trace is checked, including cells beyond 5,000.
    """
    issues: list[ValidationIssue] = []

    def report(code: str, field: str, message: str) -> None:
        issues.append(ValidationIssue(code, field, message))

    timestamps, directions, mask = trace.timestamps, trace.directions, trace.mask
    timestamps_1d = isinstance(timestamps, np.ndarray) and timestamps.ndim == 1
    directions_1d = isinstance(directions, np.ndarray) and directions.ndim == 1
    if timestamps_1d and directions_1d and timestamps.size != directions.size:
        report("length_mismatch", "cells", "Timestamp and direction lengths differ.")

    valid_length = trace.valid_length
    valid_count = _is_integer(valid_length) and valid_length >= 0
    if not valid_count:
        report("invalid_valid_length", "valid_length", "Valid length must be a nonnegative integer.")
    if (valid_count and valid_length == 0) or (timestamps_1d and timestamps.size == 0) or (
        directions_1d and directions.size == 0
    ):
        report("empty_trace", "cells", "A trace must contain at least one valid cell.")

    if mask is None:
        if valid_count and (
            (timestamps_1d and valid_length != timestamps.size)
            or (directions_1d and valid_length != directions.size)
        ):
            report("invalid_valid_length", "valid_length", "Valid length must match both unmasked channels.")
    else:
        if not isinstance(mask, np.ndarray) or mask.shape != (MAX_CELLS,) or mask.dtype.kind != "b":
            report("invalid_mask", "mask", f"The mask must be a boolean array of shape ({MAX_CELLS},).")
        elif valid_count and (
            valid_length > MAX_CELLS or not np.array_equal(mask, np.arange(MAX_CELLS) < valid_length)
        ):
            report("mask_valid_length_mismatch", "mask", "The mask must mark exactly the valid prefix.")
        if valid_count and valid_length > MAX_CELLS:
            report("invalid_valid_length", "valid_length", "Valid length exceeds fixed capacity.")

    for field, cells in (("timestamps", timestamps), ("directions", directions)):
        if not isinstance(cells, np.ndarray) or cells.ndim != 1:
            report("invalid_shape", field, f"{field} must be a one-dimensional NumPy array.")
            continue
        if mask is not None and cells.size != MAX_CELLS:
            report("invalid_fixed_length", field, f"{field} must contain {MAX_CELLS} positions with a mask.")
        if cells.dtype.kind not in "iuf":
            report("invalid_numeric_type", field, f"{field} must contain real numeric data.")
            continue
        finite = np.isfinite(cells)
        if not finite.all():
            report("nonfinite_values", field, f"{field} contains NaN or infinity.")

        prefix_size = int(valid_length) if mask is not None and valid_count else cells.size
        valid_cells = cells[:prefix_size]
        if field == "timestamps":
            if valid_cells.size and require_relative and valid_cells[0] != 0:
                report("nonrelative_timestamps", field, "The first valid timestamp must be zero.")
            if require_relative and np.any(valid_cells[finite[:prefix_size]] > np.finfo(np.float64).max):
                report("timestamp_out_of_range", field, "Relative timestamps must fit in float64.")
            if np.any(valid_cells[1:] < valid_cells[:-1]):
                report("decreasing_timestamps", field, "Valid timestamps must be nondecreasing in observed order.")
        elif np.any(finite[:prefix_size] & (valid_cells != -1) & (valid_cells != 1)):
            report("invalid_direction", field, "Valid directions must be -1 or +1.")
        if np.any(cells[prefix_size:] != 0):
            report("nonzero_padding", field, f"Padded {field} must be zero.")

    labels = trace.labels
    if (
        not isinstance(labels, tuple)
        or not labels
        or not all(_is_label(label) for label in labels)
        or len(set(labels)) != len(labels)
    ):
        report("invalid_labels", "labels", "Labels must be a nonempty tuple of distinct integers or nonblank strings.")
    for field, identifier in (("trace_id", trace.trace_id), ("source_dataset", trace.source_dataset)):
        if not isinstance(identifier, str) or not identifier.strip():
            report("invalid_identifier", field, f"{field} must be a nonblank string.")
    if not isinstance(trace.metadata, dict):
        report("invalid_metadata", "metadata", "Metadata must be a dictionary.")

    return ValidationResult(tuple(issues))


def validate_single_tab_trace(trace: CanonicalTrace) -> ValidationResult:
    """Validate all supplied cells and require one unmixed website trace."""
    result = validate_trace(trace)
    if result.is_valid and (len(trace.labels) != 1 or MERGE_PROVENANCE_KEY in trace.metadata):
        issue = ValidationIssue("not_single_tab", "trace", f"Trace {trace.trace_id!r} must represent one tab with one label.")
        return ValidationResult((issue,))
    return result
