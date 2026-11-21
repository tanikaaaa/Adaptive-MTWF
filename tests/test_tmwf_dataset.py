"""Preservation and merge checks for the existing local TMWF TBB release."""

import os
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_mtwf.data import MAX_CELLS, TraceValidationError, canonicalize_trace, merge_with_offsets, validate_trace
from adaptive_mtwf.data.loaders.tmwf import SOURCE_DATASET, load_single_tab_split
from scripts.audit_tmwf_single_tab import verify_real_merges

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "TMWF" / "dataset" / "tbb single tab"


class ReleasedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_dir = Path(os.environ.get("TMWF_DATASET_DIR", DEFAULT_DATASET_DIR))
        cls.dataset = load_single_tab_split(source_dir, "train")

    def test_monitored_trace_label_id_and_provenance(self):
        trace = self.dataset.trace(0)
        self.assertTrue(validate_trace(trace).is_valid)
        self.assertEqual(trace.labels, (0,))
        self.assertIs(type(trace.labels[0]), int)
        self.assertEqual(trace.source_dataset, SOURCE_DATASET)
        self.assertEqual(trace.trace_id, self.dataset.trace(0).trace_id)
        self.assertNotEqual(trace.trace_id, self.dataset.trace(1).trace_id)
        self.assertEqual(trace.metadata, {
            "source_file": str(self.dataset.source_file),
            "source_sha256": self.dataset.source_sha256,
            "source_index": 0,
            "split": "train",
            "original_label": 0,
        })

    def test_unmonitored_label_remains_negative_one(self):
        index = 4_000
        trace = self.dataset.trace(index)
        self.assertEqual(int(self.dataset.labels[index]), -1)
        self.assertEqual(trace.labels, (-1,))
        self.assertEqual(trace.metadata["original_label"], -1)
        self.assertTrue(validate_trace(trace).is_valid)

    def test_short_and_long_sources_preserve_all_observations_before_windowing(self):
        for index, raw_length in ((0, 937), (83, 5_833)):
            with self.subTest(index=index):
                times, directions = self.dataset.timestamps[index], self.dataset.directions[index]
                trace = self.dataset.trace(index)
                expected = times.astype(np.float64) - float(times[0])
                self.assertIsNone(trace.mask)
                self.assertEqual(trace.valid_length, raw_length)
                assert_array_equal(trace.timestamps, expected)
                assert_array_equal(trace.directions, directions)
                self.assertFalse(np.shares_memory(trace.directions, directions))
                zero_times = times == 0
                self.assertTrue(zero_times.any())
                assert_array_equal(trace.directions[zero_times], directions[zero_times])

                fixed = canonicalize_trace(trace)
                kept = min(raw_length, MAX_CELLS)
                self.assertEqual(fixed.valid_length, kept)
                assert_array_equal(fixed.mask, np.arange(MAX_CELLS) < kept)
                assert_array_equal(fixed.timestamps[fixed.mask], expected[:kept])
                assert_array_equal(fixed.directions[fixed.mask], directions[:kept])

    def test_known_decreasing_trace_is_preserved_and_flagged(self):
        index = 385
        times, directions = self.dataset.timestamps[index], self.dataset.directions[index]
        trace = self.dataset.trace(index)
        expected = times.astype(np.float64) - float(times[0])
        self.assertEqual(np.count_nonzero(np.diff(expected) < 0), 1)
        assert_array_equal(trace.timestamps, expected)
        assert_array_equal(trace.directions, directions)
        self.assertEqual(trace.valid_length, times.size)
        self.assertEqual([issue.code for issue in validate_trace(trace).issues], ["decreasing_timestamps"])
        with self.assertRaises(TraceValidationError):
            canonicalize_trace(trace)
        with self.assertRaises(TraceValidationError):
            merge_with_offsets([trace, self.dataset.trace(0)], [0.0, 0.0])

    def test_two_real_merges_reach_canonicalization_with_all_cells(self):
        with patch("adaptive_mtwf.data.merge.canonicalize_trace", wraps=canonicalize_trace) as finalize:
            result = verify_real_merges(self.dataset)

        indices = [source["source_index"] for source in result["sources"]]
        self.assertEqual(indices, [83, 0])
        self.assertEqual(finalize.call_count, 2)
        for case, call in zip(result["checks"].values(), finalize.call_args_list):
            with self.subTest(offsets=case["offsets_in_source_units"]):
                unwindowed = call.args[0]
                self.assertIsNone(unwindowed.mask)
                self.assertEqual(unwindowed.valid_length, 6_770)
                self.assertEqual(case["retained_length"], MAX_CELLS)
                self.assertEqual(case["shapes"], {channel: (MAX_CELLS,) for channel in ("timestamps", "directions", "mask")})
                self.assertEqual(case["labels"], (1, 0))
                expected = []
                for index, offset in zip(indices, case["offsets_in_source_units"]):
                    times = self.dataset.timestamps[index]
                    directions = self.dataset.directions[index]
                    expected.extend((float(time) - float(times[0]) + offset, int(direction))
                                    for time, direction in zip(times, directions))
                expected.sort(key=lambda cell: cell[0])
                assert_array_equal(unwindowed.timestamps, [cell[0] for cell in expected])
                assert_array_equal(unwindowed.directions, [cell[1] for cell in expected])


class AdapterFixtureTests(unittest.TestCase):
    def test_finite_and_nonfinite_origins_do_not_hide_other_anomalies(self):
        raw = {
            "time": [np.array([12, 11, 12], dtype=np.float32), np.array([np.nan, 5, 4], dtype=np.float32)],
            "data": [np.array([1, 0, -1], dtype=np.int8), np.array([-1, 1, -1], dtype=np.int8)],
            "label": np.array([2, -1], dtype=np.int8),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / "test").open("wb") as file:
                pickle.dump(raw, file, protocol=4)
            dataset = load_single_tab_split(path, "test")

        first = dataset.trace(0)
        assert_array_equal(first.timestamps, [0, -1, 0])
        assert_array_equal(first.directions, raw["data"][0])
        self.assertEqual({issue.code for issue in validate_trace(first).issues}, {
            "decreasing_timestamps", "invalid_direction",
        })
        second = dataset.trace(1)
        assert_array_equal(second.timestamps, raw["time"][1])
        self.assertTrue({"decreasing_timestamps", "nonfinite_values"}.issubset(
            issue.code for issue in validate_trace(second).issues
        ))
