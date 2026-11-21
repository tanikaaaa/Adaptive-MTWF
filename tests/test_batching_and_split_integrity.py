"""Exercise batch and partition checks with artificial sequences and IDs."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data.batching import DEFAULT_MAX_LENGTH, SequenceItem, collate_sequences, encode_multihot
from adaptive_mtwf.data.split_integrity import audit_split_ids
from scripts.audit_tmwf_source_integrity import record_digest


class SplitIntegrityTests(unittest.TestCase):
    def test_counts_and_pairwise_overlap_across_three_partitions(self):
        disjoint = {"train": ["a", "b", "b"], "validation": ["c"], "test": ["d"]}
        report = audit_split_ids(disjoint)
        self.assertTrue(report.is_disjoint)
        self.assertEqual(report.row_counts["train"], 3)
        self.assertEqual(report.unique_id_counts["train"], 2)
        overlapping = audit_split_ids({**disjoint, "validation": ["b", "d"]})
        self.assertFalse(overlapping.is_disjoint)
        self.assertEqual({(pair.left, pair.right): pair.shared_id_count for pair in overlapping.overlaps}, {
            ("test", "train"): 0, ("test", "validation"): 1, ("train", "validation"): 1,
        })

    def test_missing_identity_fails_instead_of_passing_as_disjoint(self):
        with self.assertRaises(ValueError):
            audit_split_ids({"train": [""], "test": ["record"]})

    def test_record_identity_checks_every_cell_and_preserves_input(self):
        first = np.array([0.0, 0.5, 0.25], dtype=np.float32)
        second = np.array([-1, 1, 1], dtype=np.int8)
        digest = record_digest(first, second)
        self.assertEqual(digest, record_digest(first.copy(), second.copy()))
        self.assertNotEqual(digest, record_digest(first[::-1], second[::-1]))
        self.assertNotEqual(digest, record_digest(first, second[:-1]))
        assert_array_equal(first, [0.0, 0.5, 0.25])
        assert_array_equal(second, [-1, 1, 1])


class SequenceBatchTests(unittest.TestCase):
    def test_ordered_vocabulary_yields_correct_float32_multi_hot_target(self):
        target = encode_multihot(("c", "a", "c"), ("b", "c", "a", "d"))
        self.assertEqual(target.shape, (4,))
        self.assertEqual(target.dtype, np.float32)
        assert_array_equal(target, [0, 1, 1, 0])
        with self.assertRaisesRegex(ValueError, "outside the supplied vocabulary"):
            encode_multihot(("background",), ("a", "b"))

    def test_dynamic_batch_preserves_valid_zeros_and_masks_padding(self):
        items = [
            SequenceItem("short", [[4, 0], [0, 2]], [1, 0, 1]),
            SequenceItem("long", [[7, 8, 9], [11, 12, 13]], [1, 1, 0]),
        ]
        result = collate_sequences(items)
        self.assertEqual(result.inputs.shape, (2, 2, 3))
        self.assertEqual(result.inputs.dtype, np.float32)
        self.assertEqual(result.valid_mask.dtype, np.bool_)
        self.assertEqual(result.targets.shape, (2, 3))
        self.assertEqual(result.sample_ids, ("short", "long"))
        assert_array_equal(result.inputs[0], [[4, 0, 0], [0, 2, 0]])
        assert_array_equal(result.inputs[1], items[1].channels)
        assert_array_equal(result.valid_mask, [[True, True, False], [True, True, True]])
        assert_array_equal(result.targets, [[1, 0, 1], [1, 1, 0]])
        assert_array_equal(result.original_lengths, [2, 3])
        assert_array_equal(result.retained_lengths, [2, 3])
        assert_array_equal(result.retained_fractions, [1.0, 1.0])
        assert_array_equal(result.truncated, [False, False])

    def test_ceiling_only_truncates_above_limit_without_changing_source(self):
        channels = np.arange(16, dtype=np.float32).reshape(2, 8)
        original = channels.copy()
        items = [SequenceItem("long", channels, [1]), SequenceItem("exact", channels[:, :4], [0])]
        result = collate_sequences(items, max_length=4)
        assert_array_equal(result.inputs[0], original[:, :4])
        assert_array_equal(result.inputs[1], original[:, :4])
        assert_array_equal(result.original_lengths, [8, 4])
        assert_array_equal(result.retained_lengths, [4, 4])
        assert_array_equal(result.retained_fractions, [0.5, 1.0])
        assert_array_equal(result.truncated, [True, False])
        assert_array_equal(result.valid_mask, True)
        assert_array_equal(channels, original)

    def test_default_ceiling_is_thirty_thousand_and_not_a_fixed_window(self):
        for length in (6_001, DEFAULT_MAX_LENGTH + 1):
            with self.subTest(length=length):
                item = SequenceItem("sequence", np.ones((2, length)), [1])
                result = collate_sequences([item])
                self.assertEqual(result.inputs.shape, (1, 2, min(length, 30_000)))
                self.assertEqual(result.truncated[0], length > 30_000)

    def test_bad_shapes_and_values_raise(self):
        example = SequenceItem("a", [[1, 2]], [0, 1])
        for bad in (replace(example, channels=[1, 2]), replace(example, channels=[[1, np.inf]]),
                    replace(example, channels=[[1, 1e200]]), replace(example, target=[0, 0.5])):
            with self.subTest(example=bad), self.assertRaises(ValueError):
                collate_sequences([bad])
        with self.assertRaises(ValueError):
            collate_sequences([example, replace(example, target=[1])])
        with self.assertRaises(ValueError):
            collate_sequences([example], max_length=False)
