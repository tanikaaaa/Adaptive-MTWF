"""Mathematical, preservation, and reproducibility checks for MTWF synthesis."""

import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data import (
    MAX_SOURCE_CELLS,
    SynthesisRejected,
    TraceValidationError,
    assess_synthesis_eligibility,
    canonicalize_trace,
    cap_single_tab_trace,
    make_trace,
    merge_variable_length,
    merge_with_offsets,
    synthesize_delay,
    synthesize_ratio,
    validate_trace,
)
from adaptive_mtwf.data.merge import _randomize_ties


def source(index, timestamps, directions=None, *, label=None):
    return make_trace(
        timestamps, np.ones(len(timestamps), dtype=np.int8) if directions is None else directions,
        labels=(index if label is None else label,), trace_id=f"source-{index}", source_dataset="artificial",
        metadata={"split": "test", "source_index": index},
    )


def details(trace):
    return trace.metadata["synthesis"]


class RatioSynthesisTests(unittest.TestCase):
    def test_two_tabs_at_ten_and_fifty_percent(self):
        traces = [source(0, [0, 2, 10]), source(1, [0, 8], [-1, -1])]
        for target, offset, overlap in ((0.1, 9.0, 1.0), (0.5, 5.0, 5.0)):
            with self.subTest(target=target):
                result = synthesize_ratio(traces, target)
                provenance = details(result)
                self.assertEqual(provenance["target_overlap_ratio"], target)
                self.assertEqual([entry["start_offset"] for entry in provenance["sources"]], [0.0, offset])
                self.assertEqual(provenance["actual_adjacent_overlaps"][0]["overlap_ratio"], target)
                self.assertEqual(provenance["actual_adjacent_overlaps"][0]["overlap_duration"], overlap)
                assert_array_equal(result.timestamps, [0, 2, offset, 10, offset + 8])
                assert_array_equal(result.directions, [1, 1, -1, 1, -1])

    def test_successor_exactly_as_long_as_required_is_accepted(self):
        result = synthesize_ratio([source(0, [0, 3]), source(1, [0, 0.3])], 0.1)
        self.assertAlmostEqual(details(result)["actual_adjacent_overlaps"][0]["overlap_ratio"], 0.1, places=12)

    def test_too_short_successor_reports_actual_overlap(self):
        traces = [source(0, [0, 10]), source(1, [0, 4])]
        with self.assertRaises(SynthesisRejected) as raised:
            synthesize_ratio(traces, 0.5)
        rejection = raised.exception.as_dict()
        self.assertEqual(rejection["code"], "successor_too_short")
        self.assertEqual(rejection["details"]["overlap_ratio"], 0.4)
        self.assertEqual(rejection["details"]["required_overlap_duration"], 5.0)
        self.assertEqual(rejection["details"]["successor_trace_id"], "source-1")

    def test_three_four_and_five_tabs_use_sequential_predecessor_durations(self):
        traces = [source(index, [0, duration]) for index, duration in enumerate([10, 12, 8, 7, 6])]
        for count in (3, 4, 5):
            with self.subTest(count=count):
                result = synthesize_ratio(traces[:count], 0.5)
                provenance = details(result)
                self.assertEqual(provenance["tab_count"], count)
                self.assertEqual([entry["start_offset"] for entry in provenance["sources"]], [0, 5, 11, 15, 18.5][:count])
                self.assertEqual([pair["overlap_ratio"] for pair in provenance["actual_adjacent_overlaps"]], [0.5] * (count - 1))
                self.assertEqual(len(provenance["actual_pairwise_overlaps"]), count * (count - 1) // 2)
                self.assertEqual(result.valid_length, count * 2)
                self.assertEqual(result.labels, tuple(range(count)))

    def test_each_source_is_capped_but_mixture_is_unpadded_and_full(self):
        a = source(0, np.arange(6_003))
        b = source(1, np.arange(5_200), np.full(5_200, -1))
        result = synthesize_ratio([a, b], 0.5)
        provenance = details(result)
        self.assertIsNone(result.mask)
        self.assertEqual(result.valid_length, 2 * MAX_SOURCE_CELLS)
        self.assertTrue(validate_trace(result).is_valid)
        self.assertEqual([entry["original_valid_length"] for entry in provenance["sources"]], [6_003, 5_200])
        self.assertEqual([entry["retained_length"] for entry in provenance["sources"]], [5_000, 5_000])
        self.assertEqual([entry["duration"] for entry in provenance["sources"]], [4_999, 4_999])
        assert_array_equal(result.timestamps[result.directions == 1], np.arange(5_000))
        assert_array_equal(result.timestamps[result.directions == -1], np.arange(5_000) + 2_499.5)
        self.assertEqual(merge_variable_length([a, b], [0, 0]).valid_length, 11_203)
        self.assertEqual(merge_with_offsets([a, b], [0, 0]).valid_length, 5_000)

    def test_already_padded_source_is_capped_using_only_valid_cells(self):
        original = source(0, [0, 4])
        padded = canonicalize_trace(original)
        capped = cap_single_tab_trace(padded)
        self.assertIsNone(capped.mask)
        assert_array_equal(capped.timestamps, original.timestamps)
        assert_array_equal(capped.directions, original.directions)
        self.assertEqual(capped.valid_length, 2)

    def test_monitored_label_union_and_full_source_provenance(self):
        traces = [source(index, [0, 10], label=label) for index, label in enumerate([4, -1, 4, 7])]
        result = synthesize_ratio(traces, 0.2, seed=42)
        provenance = details(result)
        self.assertEqual(result.labels, (4, 7))
        self.assertEqual(provenance["rng_seed"], 42)
        self.assertEqual(provenance["synthesis_mode"], "M_ratio")
        self.assertEqual(provenance["tie_policy"], "seeded_random")
        self.assertEqual(provenance["sources"][1], {
            "source_order": 1, "trace_id": "source-1", "source_dataset": "artificial", "split": "test",
            "labels": [-1], "original_valid_length": 2, "retained_length": 2, "duration": 10.0,
            "start_offset": 8.0, "timeline_start_offset": 8.0, "source_index": 1,
        })
        with self.assertRaises(SynthesisRejected) as raised:
            synthesize_ratio([traces[1], traces[1]], 0.2)
        self.assertEqual(raised.exception.code, "no_monitored_labels")

    def test_sources_remain_unchanged(self):
        traces = [source(1, [0, 2, 3], [-1, 1, -1]), source(2, [0, 3], [1, -1])]
        before = [(trace.timestamps.copy(), trace.directions.copy(), trace.metadata.copy()) for trace in traces]
        for trace in traces:
            trace.timestamps.setflags(write=False)
            trace.directions.setflags(write=False)
        synthesize_ratio(traces, 0.5)
        for trace, (timestamps, directions, metadata) in zip(traces, before):
            assert_array_equal(trace.timestamps, timestamps)
            assert_array_equal(trace.directions, directions)
            self.assertEqual(trace.metadata, metadata)

    def test_decreasing_timestamp_after_cap_is_reported_and_preserved(self):
        original = source(0, np.arange(MAX_SOURCE_CELLS + 1))
        times = original.timestamps.copy()
        times[-1] = 1
        bad = replace(original, timestamps=times)
        self.assertEqual([issue.code for issue in assess_synthesis_eligibility(bad).issues], ["decreasing_timestamps"])
        with self.assertRaises(TraceValidationError):
            cap_single_tab_trace(bad)
        traces = [bad, source(1, [0, 10_000])]
        for operation in (lambda: synthesize_ratio(traces, 0.5), lambda: synthesize_delay(traces, delays=[2])):
            with self.assertRaises(SynthesisRejected) as raised:
                operation()
            self.assertEqual(raised.exception.code, "ineligible_sources")
            self.assertEqual(raised.exception.details["exclusions"][0]["issues"][0]["code"], "decreasing_timestamps")
        self.assertEqual(bad.timestamps[-1], 1)
        self.assertEqual(bad.valid_length, 5_001)

    def test_invalid_tab_count_target_and_zero_predecessor_duration(self):
        instant = source(0, [0, 0])
        normal = source(1, [0, 2])
        for traces, ratio, code in (
            ([normal], 0.1, "invalid_tab_count"), ([normal] * 6, 0.2, "invalid_tab_count"),
            ([normal, instant], 0.25, "invalid_target_ratio"), ([instant, normal], 0.1, "zero_predecessor_duration"),
        ):
            with self.subTest(code=code), self.assertRaises(SynthesisRejected) as raised:
                synthesize_ratio(traces, ratio)
            self.assertEqual(raised.exception.code, code)

    def test_zero_duration_is_ineligible_in_every_ratio_source_position(self):
        for position in (0, 1, 2):
            with self.subTest(position=position):
                traces = [source(i, [0, 10] if i != position else [0, 0]) for i in range(3)]
                with self.assertRaises(SynthesisRejected) as raised:
                    synthesize_ratio(traces, 0.1)
                self.assertEqual(raised.exception.code,
                                 "zero_successor_duration" if position == 2 else "zero_predecessor_duration")


class DelaySynthesisTests(unittest.TestCase):
    def test_explicit_offsets_are_measured_on_the_rebased_timeline(self):
        result = synthesize_delay([source(0, [0, 8]), source(1, [0, 8])], start_offsets=[-1, -3])
        provenance = details(result)
        self.assertEqual(provenance["synthesis_mode"], "M_delay")
        self.assertIsNone(provenance["target_overlap_ratio"])
        self.assertEqual([entry["start_offset"] for entry in provenance["sources"]], [-1, -3])
        self.assertEqual([entry["timeline_start_offset"] for entry in provenance["sources"]], [2, 0])
        self.assertEqual(provenance["actual_adjacent_overlaps"][0]["overlap_ratio"], 0.75)
        assert_array_equal(result.timestamps, [0, 2, 8, 10])

    def test_explicit_delays_are_start_to_start(self):
        result = synthesize_delay([source(i, [0, 4]) for i in range(3)], delays=[2, 4])
        provenance = details(result)
        self.assertEqual([entry["start_offset"] for entry in provenance["sources"]], [0, 2, 6])
        self.assertEqual([pair["overlap_ratio"] for pair in provenance["actual_adjacent_overlaps"]], [0.5, 0.0])

    def test_seeded_delays_repeat_and_are_independent_of_tie_policy(self):
        traces = [source(i, [0, 8]) for i in range(3)]
        first = synthesize_delay(traces, delay_range=(1, 3), seed=20)
        repeated = synthesize_delay(traces, delay_range=(1, 3), seed=20)
        legacy = synthesize_delay(traces, delay_range=(1, 3), seed=20, tie_policy="legacy_tmwf")
        different = synthesize_delay(traces, delay_range=(1, 3), seed=21)
        self.assertEqual(first.trace_id, repeated.trace_id)
        self.assertEqual(first.metadata, repeated.metadata)
        self.assertEqual(first.timestamps.tobytes(), repeated.timestamps.tobytes())
        self.assertEqual(first.directions.tobytes(), repeated.directions.tobytes())
        self.assertEqual(details(first)["delay_schedule"], details(legacy)["delay_schedule"])
        self.assertNotEqual(details(first)["delay_schedule"], details(different)["delay_schedule"])

    def test_zero_duration_predecessor_has_no_defined_ratio(self):
        result = synthesize_delay([source(0, [0, 0]), source(1, [0, 10])], delays=[0])
        self.assertEqual(details(result)["actual_adjacent_overlaps"][0]["overlap_duration"], 0.0)
        self.assertIsNone(details(result)["actual_adjacent_overlaps"][0]["overlap_ratio"])

    def test_delay_schedule_must_be_unambiguous(self):
        traces = [source(0, [0, 1]), source(1, [0, 10])]
        for kwargs in ({}, {"delays": [-1]}, {"delay_range": (3, 1)}, {"delays": [1], "start_offsets": [0, 1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                synthesize_delay(traces, **kwargs)


class TiePolicyTests(unittest.TestCase):
    def test_legacy_ties_preserve_source_and_cell_order(self):
        a = source(0, [0, 1, 1, 3], [1, 1, -1, 1])
        b = source(1, [0, 0, 1], [-1, 1, -1])
        result = synthesize_delay([a, b], delays=[1], tie_policy="legacy_tmwf")
        assert_array_equal(result.timestamps, [0, 1, 1, 1, 1, 2, 3])
        assert_array_equal(result.directions, [1, 1, -1, -1, 1, -1, 1])
        self.assertEqual(details(result)["tie_policy"], "legacy_tmwf")
        variable = merge_variable_length([a, b], [0, 1], tie_policy="legacy_tmwf")
        legacy = merge_with_offsets([a, b], [0, 1])
        self.assertEqual(variable.metadata["tie_policy"], "legacy_tmwf")
        self.assertEqual(legacy.trace_id, variable.trace_id)
        self.assertEqual(legacy.metadata, {"merge_provenance": variable.metadata["merge_provenance"]})

    def test_seed_changes_only_cross_source_equal_time_order_for_fixed_offsets(self):
        a = source(0, [0, 1, 1, 1, 3], [1, 1, 1, 1, 1])
        b = source(1, [0, 0, 0, 1], [-1, -1, -1, -1])
        traces = [a, b]
        first = synthesize_delay(traces, delays=[1], seed=7)
        repeat = synthesize_delay(traces, delays=[1], seed=7)
        self.assertEqual(first.trace_id, repeat.trace_id)
        self.assertEqual(first.metadata, repeat.metadata)
        self.assertEqual(first.timestamps.tobytes(), repeat.timestamps.tobytes())
        self.assertEqual(first.directions.tobytes(), repeat.directions.tobytes())
        alternatives = [synthesize_delay(traces, delays=[1], seed=seed) for seed in range(10)]
        self.assertGreater(len({trace.directions.tobytes() for trace in alternatives}), 1)
        for trace in alternatives:
            assert_array_equal(trace.timestamps, first.timestamps)
            assert_array_equal(trace.directions[trace.timestamps != 1], first.directions[first.timestamps != 1])
            assert_array_equal(np.sort(trace.directions[trace.timestamps == 1]), [-1, -1, -1, 1, 1, 1])

    def test_each_source_keeps_its_cell_indices_inside_cross_source_ties(self):
        timestamps = np.array([0, 0, 0, 1, 0, 0, 2, 0, 0], dtype=np.float64)
        legacy = np.argsort(timestamps, kind="stable")
        source_ids = np.repeat([0, 1, 2], [4, 3, 2])
        for seed in range(10):
            order = legacy.copy()
            _randomize_ties(order, timestamps, [4, 3, 2], seed)
            assert_array_equal(timestamps[order], timestamps[legacy])
            for source_id in (0, 1, 2):
                assert_array_equal(order[source_ids[order] == source_id], np.flatnonzero(source_ids == source_id))
