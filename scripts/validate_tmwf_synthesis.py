"""Validate controlled synthesis on five deterministic clean TMWF test sources."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data import (
    MAX_SOURCE_CELLS, CanonicalTrace, assess_synthesis_eligibility, synthesize_delay, synthesize_ratio, validate_trace,
)
from adaptive_mtwf.data.loaders.tmwf import UNMONITORED_LABEL, TMWFSingleTabSplit, load_single_tab_split
from adaptive_mtwf.data.synthesis import OVERLAP_RATIO_TOLERANCE, TARGET_RATIOS

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/data_audits/tmwf_synthesis_validation.json"


def _scan_split(dataset: TMWFSingleTabSplit) -> tuple[dict, list[CanonicalTrace]]:
    exclusions: dict[str, list[int]] = defaultdict(list)
    excluded_count = 0
    selected: dict[int, CanonicalTrace] = {}
    for index in range(len(dataset)):
        trace = dataset.trace(index)
        eligibility = assess_synthesis_eligibility(trace)
        if not eligibility.is_valid:
            excluded_count += 1
            for code in sorted({issue.code for issue in eligibility.issues}):
                exclusions[code].append(index)
            continue
        label = int(trace.labels[0])
        if dataset.split == "test" and len(selected) < 5 and label != UNMONITORED_LABEL and label not in selected:
            if trace.timestamps[min(trace.valid_length, MAX_SOURCE_CELLS) - 1] > 0:
                selected[label] = trace
    report = {
        "source_file": str(dataset.source_file),
        "source_sha256": dataset.source_sha256,
        "scanned_count": len(dataset),
        "eligible_count": len(dataset) - excluded_count,
        "excluded_count": excluded_count,
        "excluded_source_indices_by_reason": dict(exclusions),
    }
    return report, list(selected.values())


def _validate_example(mixed: CanonicalTrace, sources: list[CanonicalTrace]) -> dict:
    """Verify full mixed cell contents and measure adjacent intersections."""
    validate_trace(mixed).raise_if_invalid()
    provenance = mixed.metadata["synthesis"]
    records = provenance["sources"]
    target = provenance["target_overlap_ratio"]
    starts = [record["timeline_start_offset"] for record in records]
    durations = []
    expected_events = []
    for trace, record, start in zip(sources, records, starts):
        count = min(trace.valid_length, MAX_SOURCE_CELLS)
        duration = float(trace.timestamps[count - 1]) - float(trace.timestamps[0])
        durations.append(duration)
        if record["trace_id"] != trace.trace_id or record["retained_length"] != count or record["duration"] != duration:
            raise AssertionError(f"Recorded source does not match {trace.trace_id}.")
        expected_events.extend((float(time) + start, int(direction))
                               for time, direction in zip(trace.timestamps[:count], trace.directions[:count]))
    if mixed.mask is not None or mixed.valid_length != len(expected_events):
        raise AssertionError("Mixture must contain all retained source cells as an unpadded trace.")
    if mixed.labels != tuple(dict.fromkeys(trace.labels[0] for trace in sources)):
        raise AssertionError("Mixture is missing a selected website label.")
    actual_events = zip(mixed.timestamps.tolist(), mixed.directions.tolist())
    assert_array_equal(sorted(actual_events), sorted(expected_events))

    actual_ratios = []
    for j, pair in enumerate(provenance["actual_adjacent_overlaps"]):
        if starts[j] > starts[j + 1]:
            raise AssertionError("Smoke example starts are expected to follow the selected source order.")
        # In these ordered examples, the intersection begins at the successor.
        remaining_predecessor_time = max(0, durations[j] - (starts[j + 1] - starts[j]))
        actual_duration = min(remaining_predecessor_time, durations[j + 1])
        measured_ratio = actual_duration / durations[j]
        if not math.isclose(measured_ratio, pair["overlap_ratio"], rel_tol=0, abs_tol=OVERLAP_RATIO_TOLERANCE):
            raise AssertionError(f"Reported and measured ratios differ for source pair {j}, {j + 1}.")
        if target is not None and not math.isclose(measured_ratio, target, rel_tol=0, abs_tol=OVERLAP_RATIO_TOLERANCE):
            raise AssertionError(f"Source pair {j}, {j + 1} measured {measured_ratio} for target {target}.")
        actual_ratios.append(pair["overlap_ratio"])

    print(f"{provenance['synthesis_mode']} | {provenance['tab_count']} tabs | classes {mixed.labels} | {mixed.valid_length:,} cells")
    print(f"  durations {[round(value, 6) for value in durations]}; offsets {[round(value, 6) for value in starts]}")
    print(f"  target {target}; adjacent actual ratios {[round(value, 12) for value in actual_ratios]}")
    print(f"  first five events {list(zip(mixed.timestamps[:5].tolist(), mixed.directions[:5].tolist()))}")
    result = {
        "trace_id": mixed.trace_id,
        "synthesis_mode": provenance["synthesis_mode"],
        "tab_count": provenance["tab_count"],
        "source_pool_positions": list(range(len(sources))),
        "labels": mixed.labels,
        "start_offsets": [record["start_offset"] for record in records],
        "target_overlap_ratio": target,
        "actual_adjacent_overlaps": provenance["actual_adjacent_overlaps"],
        "merged_length": mixed.valid_length,
    }
    if "delay_schedule" in provenance:
        result["delay_schedule"] = provenance["delay_schedule"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with the trusted TMWF train and test files.")
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    splits = {}
    selected = []
    for name in ("train", "test"):
        dataset = load_single_tab_split(args.dataset_dir, name)
        splits[name], selected = _scan_split(dataset)
        print(f"{name}: {splits[name]['eligible_count']:,} eligible, {splits[name]['excluded_count']} excluded")
        del dataset
    if len(selected) != 5:
        raise ValueError("Need five eligible positive-duration test sources from different monitored classes.")

    # Nondecreasing capped durations make these examples feasible for all
    # requested ratios. The corpus eligibility scan is independent of this choice.
    selected.sort(key=lambda trace: trace.timestamps[min(trace.valid_length, MAX_SOURCE_CELLS) - 1])
    print("Selected monitored sources in synthesis order:")
    for position, trace in enumerate(selected):
        print(f"  {position}. class {trace.labels[0]} | {trace.trace_id}")

    samples = []
    source_pool = []
    for count, target in [(2, target) for target in TARGET_RATIOS] + [(3, 0.3), (4, 0.3), (5, 0.3)]:
        sources = selected[:count]
        mixed = synthesize_ratio(sources, target, seed=args.seed)
        samples.append(_validate_example(mixed, sources))
        if count == 5:
            source_pool = [
                {key: value for key, value in record.items() if key not in ("start_offset", "timeline_start_offset")}
                for record in mixed.metadata["synthesis"]["sources"]
            ]

    delayed = synthesize_delay(selected[:3], delay_range=(2.0, 4.0), seed=args.seed)
    samples.append(_validate_example(delayed, selected[:3]))
    report = {
        "max_source_cells": MAX_SOURCE_CELLS,
        "overlap_ratio_tolerance": OVERLAP_RATIO_TOLERANCE,
        "eligibility_policy": "Validate all supplied source cells before capping; retain excluded rows in the corpus.",
        "eligibility_by_split": splits,
        "rng_seed": args.seed,
        "rng_algorithm": "NumPy Generator with PCG64",
        "numpy_version": np.__version__,
        "tie_policy": "seeded_random",
        "source_pool": source_pool,
        "examples": samples,
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
    print(f"Validation written to {ARTIFACT}")


if __name__ == "__main__":
    main()
