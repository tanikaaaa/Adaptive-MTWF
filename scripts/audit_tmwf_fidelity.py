"""Run the bounded, label-matched synthetic-versus-real TMWF audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from adaptive_mtwf.data import (
    MAX_SOURCE_CELLS, assess_synthesis_eligibility, synthesize_delay, synthesize_ratio, validate_trace,
)
from adaptive_mtwf.data.fidelity import TIME_BINS, compare_features, length_coverage, summarize, summarize_features, trace_features
from adaptive_mtwf.data.loaders.tmwf import (
    REAL_COLLECTIONS, REAL_UNMONITORED_LABEL, UNMONITORED_LABEL, load_real_multitab, load_single_tab_split,
)
from adaptive_mtwf.data.synthesis import TARGET_RATIOS

OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/data_audits"
PRIMARY = {2: "2tabs", 3: "3tabs", 4: "4tabs"}
DELAY_RANGE = (2.0, 9.0)
TIE_ABLATION_LIMIT = 32


def audit_real(dataset_dir):
    audit, cohorts, patterns, coverage = {}, {}, {}, {}
    for name in REAL_COLLECTIONS:
        dataset = load_real_multitab(dataset_dir, name)
        lengths = [len(directions) for directions in dataset.directions]
        issues, labels = Counter(), Counter(dataset.labels.ravel().tolist())
        raw_rows, clean_rows, comparison_rows, comparison_patterns = [], [], [], []
        backward_steps = zero_directions = nonleading_zero_times = leading_zero_traces = 0
        for index in range(len(dataset)):
            trace = dataset.trace(index)
            recorded_times = dataset.timestamps[index]
            zero_directions += int(np.count_nonzero(trace.directions == 0))
            nonleading_zero_times += int(np.count_nonzero(recorded_times[1:] == 0))
            leading_zero_traces += int(recorded_times.size > 0 and recorded_times[0] == 0)
            codes = {issue.code for issue in validate_trace(trace).issues}
            issues.update(codes)
            backward_steps += int(np.count_nonzero(recorded_times[1:] < recorded_times[:-1]))
            if codes - {"decreasing_timestamps"}:
                continue
            features = trace_features(trace)
            raw_rows.append(features)
            if codes:
                continue
            clean_rows.append(features)
            original_labels = trace.metadata["original_tab_labels"]
            if any(label != REAL_UNMONITORED_LABEL for label in original_labels):
                comparison_rows.append(features)
                comparison_patterns.append(tuple(UNMONITORED_LABEL if label == REAL_UNMONITORED_LABEL else label
                                                 for label in original_labels))
        backgrounds = np.count_nonzero(dataset.labels == REAL_UNMONITORED_LABEL, axis=1)
        coverage[name] = length_coverage(lengths)
        cohorts[name] = comparison_rows
        patterns[name] = comparison_patterns
        audit[name] = {
            "source_file": str(dataset.source_file), "source_sha256": dataset.source_sha256,
            "source_size_bytes": dataset.source_size_bytes, "tab_count": dataset.tab_count,
            "trace_count": len(dataset), "raw_length_cells_including_empty_rows": summarize(lengths),
            "distinct_raw_lengths": len(set(lengths)), "raw_label_slot_counts": dict(sorted(labels.items())),
            "background_slots_per_trace": dict(sorted(Counter(backgrounds.tolist()).items())),
            "label_rows_with_repetitions": sum(len(set(row.tolist())) < len(row) for row in dataset.labels),
            "validation_issue_trace_counts": dict(sorted(issues.items())), "negative_adjacent_deltas": backward_steps,
            "zero_directions": zero_directions, "nonleading_zero_timestamps": nonleading_zero_times,
            "traces_with_zero_first_timestamp": leading_zero_traces,
            "raw_features_including_backward_steps": summarize_features(raw_rows),
            "clean_features_all_label_compositions": summarize_features(clean_rows),
            "clean_comparison_features_with_monitored_label": summarize_features(comparison_rows),
        }
        print(f"Real {name}: {len(dataset):,} rows; {len(comparison_rows):,} comparable clean rows", flush=True)
        del dataset
    return audit, cohorts, patterns, coverage


def make_source_pool(dataset_dir):
    dataset = load_single_tab_split(dataset_dir, "train")
    sources, durations, by_label = [], [], defaultdict(list)
    excluded = Counter()
    for index in range(len(dataset)):
        trace = dataset.trace(index)
        eligibility = assess_synthesis_eligibility(trace)
        if not eligibility.is_valid:
            excluded.update({issue.code for issue in eligibility.issues})
            continue
        duration = float(trace.timestamps[min(trace.valid_length, MAX_SOURCE_CELLS) - 1])
        if duration <= 0:
            excluded["nonpositive_capped_duration"] += 1
            continue
        trace.timestamps.setflags(write=False)
        trace.directions.setflags(write=False)
        by_label[int(trace.labels[0])].append(len(sources))
        sources.append(trace)
        durations.append(duration)
    report = {
        "source_file": str(dataset.source_file), "source_sha256": dataset.source_sha256,
        "eligible_source_count": len(sources), "exclusions_by_reason": dict(sorted(excluded.items())),
    }
    return sources, np.asarray(durations), by_label, report


def plan_sources(tabs, count, rng, durations, by_label, patterns):
    """Hold labels and source identities fixed across modes and all five ratios."""
    plans, rejected = [], 0
    background = set(by_label[UNMONITORED_LABEL])
    for _ in range(count):
        labels = patterns[int(rng.integers(len(patterns)))] if patterns else None
        if labels:
            for label, multiplicity in Counter(labels).items():
                if len(by_label[label]) < multiplicity:
                    raise ValueError(f"Too few distinct single-tab sources for label {label}.")
        for _ in range(1000):
            if labels:
                chosen = []
                for label in labels:
                    pool = by_label[label]
                    index = pool[int(rng.integers(len(pool)))]
                    while index in chosen:
                        index = pool[int(rng.integers(len(pool)))]
                    chosen.append(index)
            else:
                chosen = rng.choice(durations.size, size=tabs, replace=False).tolist()
            selected_durations = durations[chosen]
            has_monitored = labels is not None or any(index not in background for index in chosen)
            if has_monitored and np.all(selected_durations[1:] >= max(TARGET_RATIOS) * selected_durations[:-1]):
                plans.append((chosen, int(rng.integers(0, 2**63, dtype=np.uint64))))
                break
            rejected += 1
        else:
            raise RuntimeError(f"Could not draw a feasible {tabs}-tab source plan for {labels!r}.")
    return plans, rejected


def cross_source_tie_fraction(traces, records):
    """Fraction of cells whose exact shifted timestamp is shared across sources."""
    shifted = [trace.timestamps[:record["retained_length"]] + record["timeline_start_offset"]
               for trace, record in zip(traces, records)]
    times = np.concatenate(shifted)
    order = np.argsort(times, kind="stable")
    times = times[order]
    sources = np.repeat(np.arange(len(shifted)), [len(values) for values in shifted])[order]
    starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
    ends = np.r_[starts[1:], len(times)]
    shared = sources[starts] != sources[ends - 1]
    return float((ends[shared] - starts[shared]).sum() / len(times))


def interval_diagnostics(records, duration):
    """Integrate active half-open source intervals over the normalized timeline."""
    starts = np.array([source["timeline_start_offset"] for source in records])
    ends = starts + [source["duration"] for source in records]
    edges = np.linspace(0, duration, TIME_BINS + 1)
    intersections = np.maximum(0, np.minimum(ends[:, None], edges[1:]) - np.maximum(starts[:, None], edges[:-1]))
    boundaries = np.unique(np.r_[starts, ends])
    active = np.count_nonzero((starts[:, None] <= boundaries[:-1]) & (ends[:, None] > boundaries[:-1]), axis=0)
    time_widths = np.diff(boundaries)
    return intersections.sum(axis=0) / np.diff(edges), {
        "peak_active_sources": int(active.max()),
        "fraction_of_time_at_least_two_active": float(time_widths[active >= 2].sum() / duration),
        "fraction_of_time_at_least_three_active": float(time_widths[active >= 3].sum() / duration),
    }


def synthesize_condition(sources, plans, ratio):
    rows, counts_by_time, diagnostic_values = [], [], defaultdict(list)
    legacy_zero_ties, seeded_zero_ties, paired_cross_source, change_deltas = [], [], [], []
    overlap_ratios = defaultdict(list)
    altered = nonadjacent = 0
    for sample_index, (indices, seed) in enumerate(plans):
        traces = [sources[index] for index in indices]
        if len({trace.trace_id for trace in traces}) != len(traces):
            raise AssertionError("Source trace repeated within a mixture.")
        if ratio is None:
            mixed = synthesize_delay(traces, delay_range=DELAY_RANGE, seed=seed)
        else:
            mixed = synthesize_ratio(traces, ratio, seed=seed)
        features = trace_features(mixed)
        rows.append(features)
        provenance = mixed.metadata["synthesis"]
        for pair_index, overlap in enumerate(provenance["actual_adjacent_overlaps"]):
            overlap_ratios[str(pair_index)].append(overlap["overlap_ratio"])
        nonadjacent += any(pair["successor_source_order"] > pair["predecessor_source_order"] + 1
                           and pair["overlap_duration"] > 0 for pair in provenance["actual_pairwise_overlaps"])
        time_counts, concurrency = interval_diagnostics(provenance["sources"], features.scalars["duration_s"])
        counts_by_time.append(time_counts)
        cross_ties = cross_source_tie_fraction(traces, provenance["sources"])
        diagnostic_values["cross_source_tied_cell_fraction"].append(cross_ties)
        for key, value in concurrency.items():
            diagnostic_values[key].append(value)

        if sample_index < TIE_ABLATION_LIMIT:
            if ratio is None:
                legacy = synthesize_delay(traces, delay_range=DELAY_RANGE, seed=seed, tie_policy="legacy_tmwf")
            else:
                legacy = synthesize_ratio(traces, ratio, seed=seed, tie_policy="legacy_tmwf")
            if not np.array_equal(mixed.timestamps, legacy.timestamps):
                raise AssertionError("Tie policy changed the timestamp sequence.")
            legacy_ties = np.count_nonzero(legacy.timestamps[1:] == legacy.timestamps[:-1]) / (legacy.valid_length - 1)
            legacy_zero_ties.append(legacy_ties)
            seeded_zero_ties.append(features.scalars["tied_adjacent_fraction"])
            paired_cross_source.append(cross_ties)
            legacy_change = np.count_nonzero(legacy.directions[1:] != legacy.directions[:-1]) / (legacy.valid_length - 1)
            change_deltas.append(features.scalars["direction_change_rate"] - legacy_change)
            altered += not np.array_equal(mixed.directions, legacy.directions)
    diagnostics = {
        "actual_adjacent_overlap_ratio_by_predecessor_position": {key: summarize(values) for key, values in overlap_ratios.items()},
        "mixture_duration_s": summarize([row.scalars["duration_s"] for row in rows]),
        "nonadjacent_overlap_count": nonadjacent, "nonadjacent_overlap_fraction": nonadjacent / len(rows),
        "mean_active_sources_by_normalized_time_bin": np.mean(counts_by_time, axis=0).tolist(),
        **{key: summarize(values) for key, values in diagnostic_values.items()},
        "tie_policy_ablation": {
            "paired_sample_count": len(change_deltas), "changed_direction_sequence_count": altered,
            "seeded_random": {"adjacent_tie_fraction": summarize(seeded_zero_ties),
                              "cross_source_tied_cell_fraction": summarize(paired_cross_source)},
            "legacy_tmwf": {"adjacent_tie_fraction": summarize(legacy_zero_ties),
                            "cross_source_tied_cell_fraction": summarize(paired_cross_source)},
            "direction_change_rate_delta_seeded_minus_legacy": summarize(change_deltas),
        },
    }
    return rows, diagnostics


def write_json(name, obj):
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    target = OUTPUT / name
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=OUTPUT, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(text)
            file.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500, help="Samples per tab count and synthesis condition.")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--skip-plots", action="store_true", help="Run without Matplotlib.")
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error("A positive sample count and nonnegative seed are required.")
    start = time.perf_counter()
    audit, cohorts, patterns, real_coverage = audit_real(args.real_dir)
    sources, durations, by_label, source_report = make_source_pool(args.single_dir)
    conditions, synthetic_coverage, plan_reports, plotted_conditions = {}, {}, {}, {}
    for tabs in (2, 3, 4, 5):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(args.seed, spawn_key=(tabs,))))
        plans, rejected = plan_sources(tabs, args.samples, rng, durations, by_label, patterns.get(PRIMARY.get(tabs)))
        plan_bytes = json.dumps([(tuple(sources[i].metadata["source_index"] for i in indices), seed)
                                 for indices, seed in plans], separators=(",", ":")).encode()
        plan_reports[str(tabs)] = {"accepted": len(plans), "rejected_identity_draws": rejected,
                                  "source_plan_sha256": hashlib.sha256(plan_bytes).hexdigest()}
        conditions[str(tabs)], synthetic_coverage[str(tabs)], plotted_conditions[tabs] = {}, {}, {}
        comparators = [name for name, count in REAL_COLLECTIONS.items() if count == tabs]
        for ratio in (*TARGET_RATIOS, None):
            name = f"ratio_{ratio:.2f}" if ratio is not None else "delay"
            rows, diagnostics = synthesize_condition(sources, plans, ratio)
            summary = summarize_features(rows)
            comparisons = {
                real_name: compare_features(cohorts[real_name], rows,
                                            audit[real_name]["clean_comparison_features_with_monitored_label"], summary)
                for real_name in comparators
            }
            conditions[str(tabs)][name] = {"target_overlap_ratio": ratio, "features": summary,
                                          "diagnostics": diagnostics, "comparisons": comparisons}
            synthetic_coverage[str(tabs)][name] = length_coverage([row.scalars["cells"] for row in rows])
            if name in ("ratio_0.10", "ratio_0.30", "ratio_0.50", "delay"):
                plotted_conditions[tabs][name] = rows
            print(f"Synthetic {tabs} tabs / {name}: {len(rows):,} traces", flush=True)

    real_report = {
        "schema": "Pickle protocol 4; time list of float32 arrays; data list of int8 +/-1 arrays; int8 per-tab label matrix, 50=background.",
        "units": "Seconds inferred from the TMWF paper and loader evidence. Original timestamps are preserved up to widening and relative origin.",
        "real_tbb_five_tabs_available": False,
        "collections": audit,
    }
    result = {
        "seed": args.seed, "numpy_version": np.__version__, "rng_algorithm": "PCG64",
        "samples_per_condition": args.samples, "synthetic_trace_count": args.samples * 24,
        "time_bins": TIME_BINS, "source_cell_cap": MAX_SOURCE_CELLS, "source_pool": source_report,
        "synthetic_plan_by_tab_count": plan_reports, "primary_real_comparators": PRIMARY,
        "delay_start_to_start_uniform_bounds_s": DELAY_RANGE, "primary_tie_policy": "seeded_random",
        "design": "Real comparison uses strict monotonic rows with at least one monitored label. Shared source plans sample their label slots for 2-4 tabs and unstratified distinct sources for synthetic-only 5 tabs; all plans are feasible at 50% overlap. The 2tabs-extra comparison is a separate sensitivity analysis.",
        "conditions": conditions,
    }
    coverage = {"policy": "All nonempty real rows; all synthetic rows. No window applied for fidelity metrics.",
                "real": real_coverage, "synthetic": synthetic_coverage, "real_five_tab_available": False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json("tmwf_real_multitab_audit.json", real_report)
    write_json("synthetic_multitab_fidelity_audit.json", result)
    write_json("sequence_length_coverage.json", coverage)
    if not args.skip_plots:
        from scripts.plot_tmwf_fidelity import make_figures

        make_figures(OUTPUT, cohorts, plotted_conditions, conditions, coverage)
    print(f"Audited {sum(item['trace_count'] for item in audit.values()):,} real rows and {args.samples * 24:,} synthetic traces in {time.perf_counter() - start:.2f} s.", flush=True)


if __name__ == "__main__":
    main()
