"""Hand-computed tests using artificial probabilities and binary labels."""

import json
import unittest
from dataclasses import asdict

import numpy as np

from adaptive_mtwf.metrics import calibrate_threshold, ranking_metrics, threshold_metrics


class MultilabelMetricsTests(unittest.TestCase):
    def test_threshold_macro_and_micro_pooling(self):
        truth = [[1, 1], [1, 0]]
        scores = [[0.9, 0.4], [0.4, 0.9]]
        result = threshold_metrics(truth, scores)
        for name, expected in {"macro_precision": 0.5, "macro_recall": 0.25, "macro_f1": 1 / 3,
                               "micro_precision": 0.5, "micro_recall": 1 / 3, "micro_f1": 0.4,
                               "subset_accuracy": 0, "mean_predicted_cardinality": 1,
                               "mean_absolute_cardinality_error": 0.5}.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(result[name], expected)

    def test_ranking_partial_credit_fixed_k_and_auc(self):
        truth = [[1, 0, 1], [0, 1, 0]]
        scores = [[0.9, 0.8, 0.7], [0.4, 0.3, 0.2]]
        result = ranking_metrics(truth, scores, k=2)
        self.assertEqual(result["a_at_true_k"], 0.25)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertEqual(result["map_at_k"], 0.5)
        self.assertAlmostEqual(result["macro_roc_auc"], 2 / 3)
        self.assertEqual(result["per_class_roc_auc"], [1, 0, 1])
        self.assertEqual(ranking_metrics(truth, scores)["map_at_k"], 0.25)

    def test_score_ties_and_undefined_auc(self):
        truth = [[1, 0, 0], [0, 1, 0]]
        scores = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        result = ranking_metrics(truth, scores, k=1)
        self.assertEqual(result["a_at_true_k"], 0.5)
        self.assertEqual(result["micro_roc_auc"], 0.5)
        self.assertEqual(result["per_class_roc_auc"], [0.5, 0.5, None])
        self.assertEqual(result["auc_defined_classes"], 2)

    def test_empty_true_rows_have_explicit_ranking_conventions(self):
        self.assertIsNone(ranking_metrics([[0, 0]], [[0.1, 0.9]])["a_at_true_k"])
        self.assertIsNone(ranking_metrics([[0, 0]], [[0.1, 0.9]])["map_at_k"])
        self.assertEqual(ranking_metrics([[0, 0]], [[0.1, 0.9]], k=1)["map_at_k"], 0)

    def test_calibration_is_reproducible_and_uses_only_supplied_arrays(self):
        truth, scores = [[1], [0]], [[0.43], [0.41]]
        result = calibrate_threshold(truth, scores)
        self.assertEqual(result.threshold, 0.43)
        self.assertEqual(result.macro_f1, 1)
        self.assertEqual(result, calibrate_threshold(truth, scores))
        self.assertEqual(json.loads(json.dumps(asdict(result)))["threshold"], 0.43)
        self.assertEqual(calibrate_threshold([[1], [0]], [[0.9], [0.1]]).threshold, 0.5)

    def test_invalid_probabilities_targets_and_k_raise(self):
        for truth, scores in (([[1]], [[np.nan]]), ([[0.5]], [[0.1]]), ([[1]], [[1.1]]), ([[1]], [[0.2, 0.6]])):
            with self.subTest(truth=truth, scores=scores), self.assertRaises(ValueError):
                threshold_metrics(truth, scores)
        with self.assertRaises(ValueError):
            ranking_metrics([[0, 1]], [[0.1, 0.2]], k=3)
