"""Check measured trace features, distance definitions, and real-data preservation."""

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from adaptive_mtwf.data import make_trace, validate_trace
from adaptive_mtwf.data.fidelity import js_divergence, ks_statistic, length_coverage, trace_features
from adaptive_mtwf.data.loaders.tmwf import load_real_multitab
from scripts.audit_tmwf_fidelity import cross_source_tie_fraction, interval_diagnostics, plan_sources


def source(times, directions):
    return make_trace(times, directions, labels=(0,), trace_id="example", source_dataset="fixture")


class FeatureTests(unittest.TestCase):
    def test_rates_timing_bursts_and_normalized_endpoint(self):
        result = trace_features(source([0, 0, 1, 2, 4], [1, 1, -1, -1, 1]), bins=4)
        features = result.scalars
        self.assertEqual(features["cells"], 5)
        self.assertEqual(features["duration_s"], 4)
        self.assertEqual(features["cells_per_s"], 1.25)
        self.assertAlmostEqual(features["outgoing_fraction"], 0.6)
        self.assertAlmostEqual(features["incoming_fraction"], 0.4)
        self.assertEqual(features["direction_change_rate"], 0.5)
        self.assertEqual(features["iat_median_s"], 1)
        self.assertEqual(features["iat_mean_s"], 1)
        self.assertAlmostEqual(features["iat_p95_s"], 1.85)
        self.assertEqual(features["zero_iat_fraction"], 0.25)
        self.assertEqual(features["tied_adjacent_fraction"], 0.25)
        self.assertEqual(features["burst_count"], 3)
        self.assertAlmostEqual(features["burst_mean_cells"], 5 / 3)
        self.assertEqual(features["burst_max_cells"], 2)
        self.assertEqual(features["burst_p95_cells"], 2)
        assert_allclose(result.time_profile, [0.4, 0.2, 0.2, 0.2])

    def test_degenerate_and_singleton_statistics(self):
        row = trace_features(source([9], [1]), bins=3)
        self.assertEqual(row.scalars["duration_s"], 0)
        self.assertTrue(np.isnan(row.scalars["cells_per_s"]))
        self.assertTrue(np.isnan(row.scalars["iat_mean_s"]))
        self.assertTrue(np.isnan(row.scalars["direction_change_rate"]))
        self.assertEqual(row.scalars["burst_count"], 1)
        assert_array_equal(row.time_profile, [1, 0, 0])

    def test_distance_statistics_with_ties_and_disjoint_support(self):
        self.assertAlmostEqual(ks_statistic([0, 0, 1], [0, 1, 1]), 1 / 3)
        self.assertEqual(ks_statistic([0, 1], [2, 3]), 1)
        self.assertIsNone(ks_statistic([np.nan], [0]))
        self.assertEqual(js_divergence([1, 0], [0, 1]), 1)
        self.assertEqual(js_divergence([0, 1], [0, 1]), 0)

    def test_length_coverage_includes_equal_size_and_excludes_empty(self):
        report = length_coverage([0, 4000, 5000, 10000])
        row = report["candidates"]["5000"]
        self.assertEqual(report["trace_count"], 3)
        self.assertAlmostEqual(row["fully_retained_pct"], 200 / 3)
        self.assertAlmostEqual(row["truncated_pct"], 100 / 3)
        self.assertEqual(row["median_fraction_retained"], 1)
        self.assertEqual(row["truncated_p10_fraction_retained"], 0.5)
        self.assertIsNone(report["candidates"]["10000"]["truncated_p25_fraction_retained"])


class RealAdapterTests(unittest.TestCase):
    def test_repeated_labels_origin_and_backward_steps_are_preserved(self):
        raw = {
            "time": [np.array([4, 7, 6], dtype=np.float32)],
            "data": [np.array([1, -1, -1], dtype=np.int8)],
            "label": np.array([[8, 50, 50]], dtype=np.int8),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "3tabs"
            original = pickle.dumps(raw, protocol=4)
            path.write_bytes(original)
            dataset = load_real_multitab(path.parent, path.name)
            trace = dataset.trace(0)
            self.assertEqual(trace.trace_id, dataset.trace(0).trace_id)
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(trace.labels, (8, 50))
        self.assertEqual(trace.metadata["original_tab_labels"], (8, 50, 50))
        self.assertEqual(trace.metadata["original_timestamp_origin"], 4)
        self.assertEqual(trace.metadata["tab_count"], 3)
        self.assertEqual(trace.metadata["source_index"], 0)
        assert_array_equal(trace.timestamps, [0, 3, 2])
        assert_array_equal(trace.directions, raw["data"][0])
        assert_array_equal(dataset.timestamps[0], raw["time"][0])
        self.assertEqual([issue.code for issue in validate_trace(trace).issues], ["decreasing_timestamps"])
        self.assertEqual(trace_features(trace).scalars["iat_mean_s"], 1)


class DiagnosticsTests(unittest.TestCase):
    def test_paired_plans_are_seeded_feasible_and_have_unique_sources(self):
        durations = np.array([4, 4, 4, 4])
        by_label = {3: [0, 1], -1: [2, 3]}
        draws = [plan_sources(3, 6, np.random.default_rng(5), durations, by_label, [(3, -1, -1)]) for _ in range(2)]
        self.assertEqual(draws[0], draws[1])
        for indices, _ in draws[0][0]:
            self.assertEqual(len(set(indices)), 3)
            self.assertIn(indices[0], by_label[3])
            self.assertTrue(all(index in by_label[-1] for index in indices[1:]))

    def test_concurrency_and_exact_cross_source_ties(self):
        rows = [source([0, 2, 2, 4], [1, 1, -1, -1]), source([0, 2], [-1, -1])]
        records = [{"timeline_start_offset": 0, "duration": 4, "retained_length": 4},
                   {"timeline_start_offset": 2, "duration": 2, "retained_length": 2}]
        counts, diagnostics = interval_diagnostics(records, 4)
        assert_allclose(counts[:10], 1)
        assert_allclose(counts[10:], 2)
        self.assertEqual(diagnostics["peak_active_sources"], 2)
        self.assertEqual(diagnostics["fraction_of_time_at_least_two_active"], 0.5)
        self.assertEqual(cross_source_tie_fraction(rows, records), 5 / 6)
