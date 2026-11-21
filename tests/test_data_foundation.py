"""Exercise data invariants using artificial cells and no external datasets."""

import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data import (
    MAX_CELLS,
    CanonicalTrace,
    TraceValidationError,
    canonicalize_trace,
    inter_arrival_times,
    make_trace,
    merge_with_offsets,
    validate_trace,
)


def artificial(timestamps=(0, 1), directions=(1, -1), *, label="a", trace_id="a") -> CanonicalTrace:
    return make_trace(
        timestamps, directions, labels=(label,), trace_id=trace_id, source_dataset="test", metadata={"notes": ["original"]}
    )


class CanonicalizationTests(unittest.TestCase):
    def test_exact_length(self):
        source = artificial(np.arange(MAX_CELLS), np.ones(MAX_CELLS))
        fixed = canonicalize_trace(source)
        self.assertEqual(fixed.valid_length, MAX_CELLS)
        assert_array_equal(fixed.timestamps, source.timestamps)
        assert_array_equal(fixed.directions, source.directions)
        assert_array_equal(fixed.mask, np.ones(MAX_CELLS, dtype=bool))
        self.assertTrue(validate_trace(fixed).is_valid)

    def test_short_trace_padding_and_channel_types(self):
        fixed = canonicalize_trace(artificial())
        self.assertEqual(fixed.valid_length, 2)
        self.assertEqual(fixed.timestamps.shape, (5_000,))
        self.assertEqual(fixed.directions.shape, (5_000,))
        self.assertEqual(fixed.mask.shape, (5_000,))
        self.assertEqual(fixed.timestamps.dtype, np.dtype(np.float64))
        self.assertEqual(fixed.directions.dtype, np.dtype(np.int8))
        self.assertEqual(fixed.mask.dtype, np.dtype(np.bool_))
        assert_array_equal(fixed.timestamps[:2], [0, 1])
        assert_array_equal(fixed.directions[:2], [1, -1])
        assert_array_equal(fixed.timestamps[2:], 0)
        assert_array_equal(fixed.directions[2:], 0)
        assert_array_equal(fixed.mask, np.arange(MAX_CELLS) < 2)
        self.assertTrue(validate_trace(fixed).is_valid)

    def test_longer_trace_truncates_in_observed_order(self):
        length = MAX_CELLS + 3
        timestamps = np.arange(length) // 2
        directions = np.tile([1, -1], (length + 1) // 2)[:length]
        trace = artificial(timestamps, directions)
        fixed = canonicalize_trace(trace)
        self.assertEqual(trace.valid_length, length)
        self.assertEqual(fixed.valid_length, MAX_CELLS)
        assert_array_equal(fixed.timestamps, timestamps[:MAX_CELLS])
        assert_array_equal(fixed.directions, directions[:MAX_CELLS])
        assert_array_equal(fixed.mask, True)

    def test_normalization_preserves_equal_times_and_input_arrays(self):
        timestamps = np.array([10, 10.75, 10.75, 11])
        directions = np.array([-1, 1, -1, 1])
        trace = artificial(timestamps, directions)
        assert_array_equal(trace.timestamps, [0, 0.75, 0.75, 1])
        assert_array_equal(trace.directions, directions)
        assert_array_equal(timestamps, [10, 10.75, 10.75, 11])
        self.assertFalse(np.shares_memory(trace.timestamps, timestamps))
        self.assertFalse(np.shares_memory(trace.directions, directions))

    def test_large_integer_origin_is_removed_before_float_conversion(self):
        for origin in (2**60, -(2**60), -1):
            with self.subTest(origin=origin):
                trace = artificial([origin, origin + 1, origin + 3], [1, 1, -1])
                assert_array_equal(trace.timestamps, [0.0, 1.0, 3.0])

    def test_fixed_length_conversion_is_idempotent(self):
        fixed = canonicalize_trace(artificial())
        converted = canonicalize_trace(fixed)
        assert_array_equal(converted.timestamps, fixed.timestamps)
        assert_array_equal(converted.directions, fixed.directions)
        assert_array_equal(converted.mask, fixed.mask)
        self.assertEqual(converted.valid_length, fixed.valid_length)

    def test_inter_arrival_times_include_ties_and_padding(self):
        trace = artificial([8, 8.5, 8.5, 10], [-1, -1, 1, -1])
        assert_array_equal(inter_arrival_times(trace), [0, 0.5, 0, 1.5])
        fixed = canonicalize_trace(trace)
        deltas = inter_arrival_times(fixed)
        self.assertEqual(deltas.shape, (MAX_CELLS,))
        assert_array_equal(deltas[fixed.mask], [0, 0.5, 0, 1.5])
        assert_array_equal(deltas[~fixed.mask], 0)
        assert_array_equal(inter_arrival_times(artificial([5], [1])), [0])


class ValidationTests(unittest.TestCase):
    def assert_invalid(self, trace, *codes):
        result = validate_trace(trace)
        self.assertFalse(result.is_valid)
        self.assertTrue(set(codes).issubset(issue.code for issue in result.issues), result.issues)
        return result

    def test_invalid_directions_are_reported_even_after_truncation_boundary(self):
        trace = artificial(np.arange(MAX_CELLS + 1), np.ones(MAX_CELLS + 1))
        for bad in (0, 2, -2, 0.5):
            with self.subTest(bad=bad):
                directions = trace.directions.astype(float)
                directions[-1] = bad
                invalid = replace(trace, directions=directions)
                result = self.assert_invalid(invalid, "invalid_direction")
                with self.assertRaises(TraceValidationError) as caught:
                    canonicalize_trace(invalid)
                self.assertEqual(caught.exception.result, result)

    def test_decreasing_timestamps_rejected_without_reordering_input(self):
        timestamps = np.array([100, 101, 100])
        with self.assertRaises(TraceValidationError) as caught:
            artificial(timestamps, [1, -1, 1])
        self.assertIn("decreasing_timestamps", [issue.code for issue in caught.exception.result.issues])
        assert_array_equal(timestamps, [100, 101, 100])

    def test_decreasing_timestamps_detected_before_float_conversion(self):
        origin = 2**60
        with self.assertRaises(TraceValidationError) as caught:
            artificial([origin, origin + 1, origin], [1, -1, 1])
        self.assertIn("decreasing_timestamps", [issue.code for issue in caught.exception.result.issues])

    def test_nonfinite_values_are_reported_for_both_channels(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                invalid = replace(artificial(), timestamps=np.array([0, bad]), directions=np.array([1, bad]))
                result = self.assert_invalid(invalid, "nonfinite_values")
                fields = {issue.field for issue in result.issues if issue.code == "nonfinite_values"}
                self.assertEqual(fields, {"timestamps", "directions"})

    def test_overflow_during_normalization_is_rejected(self):
        limit = np.finfo(np.float64).max
        with self.assertRaises(TraceValidationError) as caught:
            artificial([-limit, limit], [1, -1])
        self.assertIn("nonfinite_values", [issue.code for issue in caught.exception.result.issues])

    def test_reports_length_mismatch_and_invalid_labels_together(self):
        trace = replace(artificial(), directions=np.array([1]), labels=("",))
        self.assert_invalid(trace, "length_mismatch", "invalid_labels", "invalid_valid_length")

    def test_invalid_label_types_and_duplicates(self):
        for labels in ((), ("  ",), (True,), (3.0,), (None,), ("a", "a"), "abc"):
            with self.subTest(labels=labels):
                self.assert_invalid(replace(artificial(), labels=labels), "invalid_labels")
        self.assertTrue(validate_trace(replace(artificial(), labels=(0, "a", 27))).is_valid)

    def test_empty_trace(self):
        with self.assertRaises(TraceValidationError) as caught:
            artificial([], [])
        self.assertIn("empty_trace", [issue.code for issue in caught.exception.result.issues])

    def test_fixed_mask_must_be_boolean_and_match_the_valid_prefix(self):
        fixed = canonicalize_trace(artificial())
        mask = fixed.mask.copy()
        mask[1], mask[2] = False, True
        self.assert_invalid(replace(fixed, mask=mask), "mask_valid_length_mismatch")
        self.assert_invalid(replace(fixed, valid_length=1), "mask_valid_length_mismatch")
        self.assert_invalid(replace(fixed, valid_length=MAX_CELLS + 1), "invalid_valid_length")
        self.assert_invalid(replace(fixed, mask=fixed.mask.astype(int)), "invalid_mask")
        self.assert_invalid(replace(fixed, mask=fixed.mask[:-1]), "invalid_mask")

    def test_fixed_channels_must_be_zero_in_padding(self):
        fixed = canonicalize_trace(artificial())
        timestamps, directions = fixed.timestamps.copy(), fixed.directions.copy()
        timestamps[-1], directions[-1] = 3, -1
        result = self.assert_invalid(replace(fixed, timestamps=timestamps, directions=directions), "nonzero_padding")
        self.assertEqual({issue.field for issue in result.issues}, {"timestamps", "directions"})


class MergeTests(unittest.TestCase):
    def assert_cells(self, trace, timestamps, directions):
        self.assertEqual(trace.valid_length, len(timestamps))
        assert_array_equal(trace.timestamps[trace.mask], timestamps)
        assert_array_equal(trace.directions[trace.mask], directions)
        self.assertTrue(validate_trace(trace).is_valid)

    def test_zero_offsets_preserve_trace_and_cell_order_at_ties(self):
        first = artificial([0, 0, 1], [1, -1, 1])
        second = artificial([0, 0, 1], [-1, 1, -1], label="b", trace_id="b")
        merged = merge_with_offsets([first, second], [0, 0])
        self.assert_cells(merged, [0, 0, 0, 0, 1, 1], [1, -1, -1, 1, 1, -1])
        repeated = merge_with_offsets([first, second], [0, 0])
        assert_array_equal(repeated.directions, merged.directions)
        self.assertEqual(merged.trace_id, repeated.trace_id)
        self.assertNotEqual(merged.trace_id, merge_with_offsets([second, first], [0, 0]).trace_id)

    def test_delayed_second_tab(self):
        first = artificial([10, 12], [1, -1])
        second = artificial([90, 91], [-1, 1], label="b", trace_id="b")
        merged = merge_with_offsets([first, second], [0, 1.5])
        self.assert_cells(merged, [0, 1.5, 2, 2.5], [1, -1, -1, 1])

    def test_offsets_are_relative_to_earliest_start(self):
        first, second = artificial(), artificial([0], [-1], label="b", trace_id="b")
        self.assert_cells(merge_with_offsets([first, second], [-2, -3]), [0, 1, 2], [-1, 1, -1])
        self.assert_cells(merge_with_offsets([first, second], [1e18, 1e18]), [0, 0, 1], [1, -1, -1])

    def test_large_integer_offsets_preserve_cell_gaps_and_provenance(self):
        first, second = artificial(), artificial([0], [-1], label="b", trace_id="b")
        offsets = [2**60, 2**60 + 1]
        merged = merge_with_offsets([first, second], offsets)
        self.assert_cells(merged, [0, 1, 1], [1, -1, -1])
        self.assertEqual([entry["start_offset"] for entry in merged.metadata["merge_provenance"]], offsets)

    def test_three_tabs_union_labels_and_keep_ordered_provenance(self):
        first = artificial([0, 3], [1, 1], label=9, trace_id="source-a")
        second = replace(artificial([0, 1], [-1, 1], label="b", trace_id="source-b"), source_dataset="other")
        third = artificial([0], [-1], label=9, trace_id="source-c")
        merged = merge_with_offsets([first, second, third], [0, 1, 2])
        self.assert_cells(merged, [0, 1, 2, 2, 3], [1, -1, 1, -1, 1])
        self.assertEqual(merged.labels, (9, "b"))
        self.assertEqual(merged.source_dataset, "synthetic")
        self.assertEqual(merged.metadata["merge_provenance"], [
            {"trace_id": "source-a", "source_dataset": "test", "start_offset": 0.0, "valid_length": 2},
            {"trace_id": "source-b", "source_dataset": "other", "start_offset": 1.0, "valid_length": 2},
            {"trace_id": "source-c", "source_dataset": "test", "start_offset": 2.0, "valid_length": 1},
        ])

    def test_merge_truncates_after_combining_and_keeps_all_labels(self):
        first = artificial(np.arange(MAX_CELLS + 3), np.ones(MAX_CELLS + 3))
        second = artificial([0], [-1], label="b", trace_id="b")
        third = artificial([0], [-1], label="c", trace_id="c")
        merged = merge_with_offsets([first, second, third], [0, 0.5, MAX_CELLS + 10])
        self.assert_cells(merged, np.insert(np.arange(MAX_CELLS - 1, dtype=float), 1, 0.5),
                          np.insert(np.ones(MAX_CELLS - 1), 1, -1))
        self.assertEqual(merged.labels, ("a", "b", "c"))
        self.assertEqual(merged.metadata["merge_provenance"][0]["valid_length"], MAX_CELLS + 3)

    def test_merge_excludes_padding_and_leaves_inputs_unchanged(self):
        first = canonicalize_trace(artificial())
        second = artificial([0], [-1], label="b", trace_id="b")
        for trace in (first, second):
            trace.timestamps.setflags(write=False)
            trace.directions.setflags(write=False)
        first.mask.setflags(write=False)

        merged = merge_with_offsets([first, second], [0, 0.5])
        self.assert_cells(merged, [0, 0.5, 1], [1, -1, -1])
        assert_array_equal(first.timestamps[:2], [0, 1])
        assert_array_equal(first.directions[:2], [1, -1])
        assert_array_equal(first.mask, np.arange(MAX_CELLS) < 2)
        assert_array_equal(second.timestamps, [0])
        assert_array_equal(second.directions, [-1])
        self.assertEqual(first.metadata, {"notes": ["original"]})
        self.assertEqual(second.metadata, {"notes": ["original"]})

    def test_invalid_sources_and_offsets_are_rejected(self):
        first, second = artificial(), artificial(trace_id="other")
        with self.assertRaises(ValueError):
            merge_with_offsets([first], [0])
        for offsets in ([0], [0, np.inf], [0, np.nan], [0, "one"], [[0], [1]]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                merge_with_offsets([first, second], offsets)
        with self.assertRaises(TraceValidationError):
            merge_with_offsets([replace(first, directions=np.array([1, 0])), second], [0, 0])
        with self.assertRaises(ValueError):
            merge_with_offsets([replace(first, labels=("a", "b")), second], [0, 0])
        same_label_merge = merge_with_offsets([first, second], [0, 0])
        with self.assertRaises(ValueError):
            merge_with_offsets([same_label_merge, second], [0, 0])
