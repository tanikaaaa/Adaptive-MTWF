"""Controlled duration-ratio and explicit-delay synthesis of single-tab traces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import accumulate, combinations
from numbers import Real
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .canonicalize import cap_single_tab_trace
from .merge import TiePolicy, _prepare_start_offsets, _validate_seed, merge_variable_length
from .schema import MAX_SOURCE_CELLS, CanonicalTrace, WebsiteLabel
from .validate import TraceValidationError, ValidationResult, validate_single_tab_trace

TARGET_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)
OVERLAP_RATIO_TOLERANCE = 1e-12


class SynthesisRejected(ValueError):
    """A rejected candidate, with a reason code and audit-ready details."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def assess_synthesis_eligibility(trace: CanonicalTrace) -> ValidationResult:
    """Report per-source exclusions without changing the trace or dataset.

    The default policy requires canonical single-tab observations with
    nondecreasing source times. It inspects even cells past the synthesis cap.
    Candidate-specific duration and label constraints are evaluated separately.
    """
    return validate_single_tab_trace(trace)


@dataclass(frozen=True)
class _Sources:
    original: Sequence[CanonicalTrace]
    capped: tuple[CanonicalTrace, ...]
    durations: tuple[float, ...]
    labels: tuple[WebsiteLabel, ...]


def _prepare_sources(traces: Sequence[CanonicalTrace]) -> _Sources:
    if not 2 <= len(traces) <= 5:
        raise SynthesisRejected("invalid_tab_count", "Synthesis requires two through five tabs.", tab_count=len(traces))
    exclusions = []
    capped = []
    for index, trace in enumerate(traces):
        try:
            capped.append(cap_single_tab_trace(trace))
        except TraceValidationError as error:
            exclusions.append({
                "source_order": index,
                "trace_id": trace.trace_id,
                "issues": [asdict(issue) for issue in error.result.issues],
            })
    if exclusions:
        raise SynthesisRejected("ineligible_sources", "Candidate contains ineligible source traces.", exclusions=exclusions)

    # The current closed-world target requires at least one monitored label.
    # The shared unmonitored label (-1) is retained in source provenance.
    source_labels = [trace.labels[0] for trace in traces]
    labels = tuple(dict.fromkeys(
        label if isinstance(label, str) else int(label) for label in source_labels if label != -1
    ))
    if not labels:
        raise SynthesisRejected("no_monitored_labels", "The current closed-world target requires at least one monitored label.")
    durations = tuple(float(trace.timestamps[-1]) - float(trace.timestamps[0]) for trace in capped)
    return _Sources(tuple(traces), tuple(capped), durations, labels)


def _finish_synthesis(
    sources: _Sources,
    offsets: ArrayLike,
    *,
    mode: str,
    tie_policy: TiePolicy,
    seed: int,
    target_ratio: Optional[float] = None,
    delay_schedule: Optional[dict[str, Any]] = None,
) -> CanonicalTrace:
    seed = _validate_seed(seed)
    supplied_offsets, relative_offsets = _prepare_start_offsets(offsets, len(sources.capped))
    starts = relative_offsets.tolist()
    ends = [start + duration for start, duration in zip(starts, sources.durations)]
    if not all(math.isfinite(end) for end in ends):
        raise SynthesisRejected("timeline_overflow", "Shifted source endpoints must fit in float64.")

    overlaps = []
    adjacent_overlaps = []
    for predecessor, successor in combinations(range(len(starts)), 2):
        duration, next_duration = sources.durations[predecessor], sources.durations[successor]
        overlap = max(0.0, min(ends[predecessor], ends[successor]) - max(starts[predecessor], starts[successor]))
        ratio = overlap / duration if duration else None
        pair = {
            "predecessor_source_order": predecessor,
            "successor_source_order": successor,
            "overlap_duration": overlap,
            "overlap_ratio": ratio,
        }
        overlaps.append(pair)
        if successor != predecessor + 1:
            continue
        adjacent_overlaps.append(pair)
        if target_ratio is None:
            continue
        too_short = next_duration < (target_ratio - OVERLAP_RATIO_TOLERANCE) * duration
        if too_short or ratio is None or not math.isclose(ratio, target_ratio, rel_tol=0, abs_tol=OVERLAP_RATIO_TOLERANCE):
            raise SynthesisRejected(
                "successor_too_short" if too_short else "overlap_not_achievable",
                "Adjacent pair cannot achieve the requested duration overlap.",
                **pair,
                target_ratio=target_ratio,
                required_overlap_duration=target_ratio * duration,
                predecessor_duration=duration,
                successor_duration=next_duration,
                predecessor_trace_id=sources.original[predecessor].trace_id,
                successor_trace_id=sources.original[successor].trace_id,
            )

    records = []
    for index, (source, capped) in enumerate(zip(sources.original, sources.capped)):
        label = source.labels[0]
        records.append({
            "source_order": index,
            "trace_id": source.trace_id,
            "source_dataset": source.source_dataset,
            "split": source.metadata.get("split"),
            "labels": [label if isinstance(label, str) else int(label)],
            "original_valid_length": int(source.valid_length),
            "retained_length": int(capped.valid_length),
            "duration": sources.durations[index],
            "start_offset": supplied_offsets[index],
            "timeline_start_offset": starts[index],
            **{key: source.metadata[key] for key in ("source_file", "source_sha256", "source_index") if key in source.metadata},
        })
    provenance = {
        "synthesis_mode": mode,
        "target_overlap_ratio": target_ratio,
        "actual_adjacent_overlaps": adjacent_overlaps,
        "actual_pairwise_overlaps": overlaps,
        "tab_count": len(sources.capped),
        "max_source_cells": MAX_SOURCE_CELLS,
        "sources": records,
        "tie_policy": tie_policy,
        "rng_seed": seed,
        "rng_algorithm": "NumPy Generator with PCG64",
        "numpy_version": np.__version__,
    }
    if delay_schedule is not None:
        provenance["delay_schedule"] = delay_schedule

    merged = merge_variable_length(sources.capped, supplied_offsets, tie_policy=tie_policy, seed=seed)
    identity = json.dumps([merged.trace_id, mode, target_ratio, tie_policy, seed, delay_schedule], sort_keys=True)
    return replace(
        merged,
        labels=sources.labels,
        trace_id=f"synthesized-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}",
        metadata={**merged.metadata, "synthesis": provenance},
    )


def synthesize_ratio(
    traces: Sequence[CanonicalTrace],
    target_ratio: float,
    *,
    seed: int = 0,
    tie_policy: TiePolicy = "seeded_random",
) -> CanonicalTrace:
    """Synthesize 2–5 tabs at one requested predecessor-duration ratio.

    Source duration spans the first and last retained cells. The successor
    starts at the predecessor's start plus (1 - target_ratio) * duration.
    Each interval intersection is measured on the resulting timeline and
    divided by the predecessor duration. The absolute ratio tolerance is
    1e-12. An infeasible pair rejects the whole candidate.
    """
    if not isinstance(target_ratio, Real) or float(target_ratio) not in TARGET_RATIOS:
        raise SynthesisRejected("invalid_target_ratio", f"Choose a target ratio from {TARGET_RATIOS}.")
    target_ratio = float(target_ratio)
    sources = _prepare_sources(traces)
    for index, duration in enumerate(sources.durations[:-1]):
        if duration <= 0:
            raise SynthesisRejected(
                "zero_predecessor_duration",
                "The requested ratio is undefined for a zero-duration predecessor.",
                predecessor_source_order=index,
                predecessor_trace_id=sources.original[index].trace_id,
            )
    if sources.durations[-1] <= 0:
        raise SynthesisRejected(
            "zero_successor_duration",
            "Controlled ratio synthesis requires every source duration to be positive.",
            source_order=len(sources.durations) - 1,
            trace_id=sources.original[-1].trace_id,
        )
    offsets = list(accumulate((1 - target_ratio) * duration for duration in sources.durations[:-1]))
    return _finish_synthesis(
        sources, [0.0, *offsets], mode="M_ratio", target_ratio=target_ratio, tie_policy=tie_policy, seed=seed,
    )


def _nonnegative_values(values: ArrayLike, count: int, name: str) -> list[float]:
    array = np.asarray(values)
    if array.shape != (count,) or array.dtype.kind not in "iuf":
        raise ValueError(f"Provide {count} real values for {name}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return array.tolist()


def synthesize_delay(
    traces: Sequence[CanonicalTrace],
    *,
    start_offsets: Optional[ArrayLike] = None,
    delays: Optional[ArrayLike] = None,
    delay_range: Optional[tuple[float, float]] = None,
    seed: int = 0,
    tie_policy: TiePolicy = "seeded_random",
) -> CanonicalTrace:
    """Synthesize from absolute start offsets or successive start-to-start delays.

    Supply exactly one schedule: K explicit start offsets, K-1 nonnegative
    delays, or uniform nonnegative delay bounds. Uniform delays use a PCG64
    stream separate from tie randomization. Overlaps are measured without a
    target. A zero-duration predecessor has an undefined (None) ratio.
    """
    if sum(value is not None for value in (start_offsets, delays, delay_range)) != 1:
        raise ValueError("Supply exactly one of start_offsets, delays, or delay_range.")
    seed = _validate_seed(seed)
    sources = _prepare_sources(traces)
    if start_offsets is not None:
        offsets = start_offsets
        schedule: dict[str, Any] = {"kind": "explicit_start_offsets"}
    else:
        count = len(sources.capped) - 1
        if delays is not None:
            gaps = _nonnegative_values(delays, count, "delays")
            schedule = {"kind": "explicit_delays", "delays": gaps}
        else:
            low, high = _nonnegative_values(delay_range, 2, "delay_range")
            if high < low:
                raise ValueError("delay_range must be in ascending order.")
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(1,))))
            gaps = rng.uniform(low, high, count).tolist()
            schedule = {"kind": "uniform_delays", "range": [low, high], "delays": gaps}
        offsets = [0.0, *accumulate(gaps)]
    return _finish_synthesis(sources, offsets, mode="M_delay", delay_schedule=schedule, tie_policy=tie_policy, seed=seed)
