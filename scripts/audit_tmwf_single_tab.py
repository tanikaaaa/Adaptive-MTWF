"""Audit both local TMWF single-tab partitions and verify two example merges."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data import MAX_CELLS, CanonicalTrace, canonicalize_trace, merge_with_offsets, validate_trace
from adaptive_mtwf.data.loaders.tmwf import UNMONITORED_LABEL, TMWFSingleTabSplit, load_single_tab_split

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/data_audits/tmwf_single_tab_audit.json"


def _distribution(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "min": float(min(values)),
        **dict(zip(("p05", "p25", "median", "p75", "p95"), percentiles.tolist())),
        "mean": float(np.mean(values)),
        "max": float(max(values)),
    }


def audit_split(dataset: TMWFSingleTabSplit) -> tuple[dict, list[int], list[float]]:
    """Count raw observations and check exact preservation across the adapter."""
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    lengths: list[int] = []
    negative_deltas: list[float] = []
    final_timestamps = []
    affected_labels: Counter[int] = Counter()
    class_counts = Counter(dataset.labels.tolist())

    for index, (times, directions) in enumerate(zip(dataset.timestamps, dataset.directions)):
        lengths.append(times.size)
        counts["empty_traces"] += int(times.size == 0 or directions.size == 0)
        counts["direction_timestamp_length_mismatch_traces"] += times.size != directions.size
        counts["nan_timestamps"] += int(np.count_nonzero(np.isnan(times)))
        counts["infinite_timestamps"] += int(np.count_nonzero(np.isinf(times)))
        counts["negative_directions"] += int(np.count_nonzero(directions == -1))
        counts["positive_directions"] += int(np.count_nonzero(directions == 1))
        counts["zero_directions"] += int(np.count_nonzero(directions == 0))
        counts["invalid_directions"] += int(np.count_nonzero((directions != -1) & (directions != 1)))
        counts["nonleading_zero_timestamps"] += int(np.count_nonzero(times[1:] == 0))
        if times.size:
            counts["traces_starting_at_zero"] += int(times[0] == 0)
            counts["traces_ending_at_zero"] += int(times[-1] == 0)
            if np.isfinite(times[-1]):
                final_timestamps.append(float(times[-1]))

        widened = times.astype(np.float64)
        finite = np.isfinite(widened)
        # Only adjacent finite pairs contribute to negative or zero deltas.
        adjacent_finite = finite[1:] & finite[:-1]
        differences = widened[1:][adjacent_finite] - widened[:-1][adjacent_finite]
        negative = differences[differences < 0]
        counts["zero_timestamp_deltas"] += int(np.count_nonzero(differences == 0))
        if negative.size:
            counts["traces_with_decreasing_timestamps"] += 1
            affected_labels[int(dataset.labels[index])] += 1
            negative_deltas.extend(negative.tolist())

        trace = dataset.trace(index)
        validation = validate_trace(trace)
        codes = {issue.code for issue in validation.issues}
        issues.update(codes)
        if finite.all() and ("decreasing_timestamps" in codes) != bool(negative.size):
            raise AssertionError(f"{dataset.split}[{index}]: validator disagrees with raw adjacent comparisons.")
        expected_times = widened - float(times[0]) if times.size and finite[0] else widened
        assert_array_equal(trace.timestamps, expected_times, err_msg=trace.trace_id)
        assert_array_equal(trace.directions, directions, err_msg=trace.trace_id)
        if trace.labels != (int(dataset.labels[index]),) or trace.valid_length != times.size or trace.mask is not None:
            raise AssertionError(f"{trace.trace_id}: label, length, or unpadded representation changed.")
        if validation.is_valid:
            fixed = canonicalize_trace(trace)
            assert_array_equal(fixed.mask, np.arange(MAX_CELLS) < times.size, err_msg=trace.trace_id)
            assert_array_equal(fixed.timestamps[fixed.mask], expected_times[:MAX_CELLS], err_msg=trace.trace_id)
            assert_array_equal(fixed.directions[fixed.mask], directions[:MAX_CELLS], err_msg=trace.trace_id)
            counts["verified_clean_fixed_prefixes"] += 1
            counts["clean_needing_padding"] += times.size < MAX_CELLS
            counts["clean_needing_truncation"] += times.size > MAX_CELLS

    length_array = np.asarray(lengths, dtype=np.int64)
    report = {
        "source_file": str(dataset.source_file),
        "source_sha256": dataset.source_sha256,
        "source_size_bytes": dataset.source_size_bytes,
        "trace_count": len(dataset),
        "class_counts": dict(sorted(class_counts.items())),
        "monitored_traces": len(dataset) - class_counts[UNMONITORED_LABEL],
        "unmonitored_traces": class_counts[UNMONITORED_LABEL],
        "total_timestamp_observations": sum(lengths),
        "raw_length_cells": _distribution(lengths),
        "traces_below_5000": int(np.count_nonzero(length_array < MAX_CELLS)),
        "traces_at_5000": int(np.count_nonzero(length_array == MAX_CELLS)),
        "traces_above_5000": int(np.count_nonzero(length_array > MAX_CELLS)),
        "empty_traces": counts["empty_traces"],
        "direction_timestamp_length_mismatch_traces": counts["direction_timestamp_length_mismatch_traces"],
        "nonfinite_values": {
            "timestamps": {"nan": counts["nan_timestamps"], "inf": counts["infinite_timestamps"]},
            "directions": {"nan": 0, "inf": 0},
        },
        "direction_counts": {"-1": counts["negative_directions"], "0": counts["zero_directions"],
                             "+1": counts["positive_directions"]},
        "invalid_direction_values": counts["invalid_directions"],
        "timestamp_observations": {
            "traces_starting_at_zero": counts["traces_starting_at_zero"],
            "traces_ending_at_zero": counts["traces_ending_at_zero"],
            "nonleading_zeros": counts["nonleading_zero_timestamps"],
            "equal_adjacent_values": counts["zero_timestamp_deltas"],
            "final_timestamp_distribution": _distribution(final_timestamps),
        },
        "negative_adjacent_timestamp_deltas": {
            "traces_affected": counts["traces_with_decreasing_timestamps"],
            "traces_affected_by_class": dict(sorted(affected_labels.items())),
            "total_count": len(negative_deltas),
            "worst_delta": min(negative_deltas, default=None),
            "absolute_magnitude": _distribution([-delta for delta in negative_deltas]),
        },
        "validator_trace_counts_by_issue": dict(sorted(issues.items())),
        "round_trip_verified_full_traces": len(dataset),
        "round_trip_verified_clean_fixed_prefixes": counts["verified_clean_fixed_prefixes"],
        "clean_traces_needing_padding": counts["clean_needing_padding"],
        "clean_traces_needing_truncation": counts["clean_needing_truncation"],
    }
    return report, lengths, negative_deltas


def _select_merge_inputs(dataset: TMWFSingleTabSplit) -> tuple[CanonicalTrace, CanonicalTrace]:
    """Select a clean monitored trace above 5,000 cells and a different class."""
    pool = [index for index, label in enumerate(dataset.labels) if label != UNMONITORED_LABEL]
    for index in pool:
        if dataset.timestamps[index].size > MAX_CELLS:
            first = dataset.trace(index)
            if validate_trace(first).is_valid:
                break
    else:
        raise ValueError("No valid monitored source above 5,000 cells was found.")
    for index in pool:
        if int(dataset.labels[index]) != first.labels[0]:
            second = dataset.trace(index)
            if validate_trace(second).is_valid:
                return first, second
    raise ValueError("No valid monitored source from a different class was found.")


def verify_real_merges(dataset: TMWFSingleTabSplit, *, show_prefix: bool = False) -> dict:
    """Compare the retained cells against an independent full Python stable sort."""
    traces = _select_merge_inputs(dataset)
    snapshots = [(trace.timestamps.copy(), trace.directions.copy(), deepcopy(trace.metadata)) for trace in traces]
    checks = {}
    for name, offsets in (("zero_offset", (0.0, 0.0)), ("delayed_start", (0.0, 2.0))):
        full_cells = [
            (float(time) + offset, int(direction))
            for trace, offset in zip(traces, offsets)
            for time, direction in zip(trace.timestamps, trace.directions)
        ]
        full_cells.sort(key=lambda cell: cell[0])
        expected = full_cells[:MAX_CELLS]
        merged = merge_with_offsets(traces, offsets)
        validate_trace(merged).raise_if_invalid()
        assert_array_equal(merged.mask, np.arange(MAX_CELLS) < len(expected))
        assert_array_equal(merged.timestamps[merged.mask], [cell[0] for cell in expected])
        assert_array_equal(merged.directions[merged.mask], [cell[1] for cell in expected])
        labels = tuple(trace.labels[0] for trace in traces)
        provenance = [
            {"trace_id": trace.trace_id, "source_dataset": trace.source_dataset,
             "start_offset": offset, "valid_length": trace.valid_length}
            for trace, offset in zip(traces, offsets)
        ]
        if merged.labels != labels or merged.metadata.get("merge_provenance") != provenance:
            raise AssertionError(f"{name}: merged labels or provenance differ from the selected source traces.")
        for trace, (times, directions, metadata) in zip(traces, snapshots):
            assert_array_equal(trace.timestamps, times)
            assert_array_equal(trace.directions, directions)
            if trace.metadata != metadata:
                raise AssertionError(f"{name}: source provenance changed.")
        checks[name] = {
            "offsets_in_source_units": offsets,
            "full_reference_length": len(full_cells),
            "retained_length": merged.valid_length,
            "shapes": {"timestamps": merged.timestamps.shape, "directions": merged.directions.shape,
                       "mask": merged.mask.shape},
            "labels": merged.labels,
            "provenance": provenance,
            "stable_reference_and_input_preservation_verified": True,
        }
        if show_prefix:
            print(f"{name} {offsets} — first eight (relative timestamp, direction) pairs:")
            print(list(zip(merged.timestamps[:8].tolist(), merged.directions[:8].tolist())))
    return {"sources": [trace.metadata for trace in traces], "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing the trusted train and test pickles.")
    args = parser.parse_args()

    splits: dict[str, dict] = {}
    all_lengths: list[int] = []
    all_negative_deltas: list[float] = []
    for name in ("train", "test"):
        dataset = load_single_tab_split(args.dataset_dir, name)
        splits[name], lengths, negative_deltas = audit_split(dataset)
        all_lengths.extend(lengths)
        all_negative_deltas.extend(negative_deltas)
        print(f"{name}: {len(dataset):,} traces audited ({dataset.source_file})")
        if name == "test":
            merge_checks = verify_real_merges(dataset, show_prefix=True)
        del dataset

    report = {
        "schema": {
            "serialization": "Python pickle protocol 4",
            "time": "list of 1D float32 timestamp arrays",
            "data": "list of 1D int8 direction arrays",
            "label": "1D int8 array of scalar labels (-1 unmonitored, 0-49 monitored)",
            "time_units": "Seconds inferred from TMWF Appendix H (p. 20) and direct offset addition in merge_with_delaytime. No unit conversion applied.",
            "direction_semantics": "+1 client-to-site, -1 site-to-client (TMWF Appendix G, p. 19).",
            "length_thresholds": "Raw timestamp row lengths; below/above 5000 correspond to padding/truncation for clean traces.",
            "negative_deltas": "Computed in float64 between adjacent finite source values in their stored order.",
        },
        "total_traces": len(all_lengths),
        "combined_raw_length_cells": _distribution(all_lengths),
        "combined_negative_delta_magnitude": _distribution([-delta for delta in all_negative_deltas]),
        "splits": splits,
        "real_merge_verification": merge_checks,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=ARTIFACT.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(serialized)
            file.close()
            temporary.replace(ARTIFACT)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Audit written to {ARTIFACT}")


if __name__ == "__main__":
    main()
