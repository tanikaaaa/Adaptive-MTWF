"""Verify protocol helpers and capacity probing with artificial values only."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from adaptive_mtwf.experiment_metadata import (
    ExperimentResult, derive_seed, fit_validation_threshold, positive_weights, reproducibility_fingerprint,
)
from scripts.probe_sequence_cnn_capacity import CONFIG, probe_step, run_probe, validate_config


class MetadataTests(unittest.TestCase):
    def test_seeds_are_stable_and_separated_by_namespace(self):
        value = derive_seed(2023, "python")
        self.assertEqual(value, derive_seed(2023, "python"))
        self.assertNotEqual(value, derive_seed(2023, "numpy"))
        self.assertNotEqual(value, derive_seed(2024, "python"))
        with self.assertRaises(ValueError):
            derive_seed(True, "python")

    def test_weights_use_explicit_counts_with_zero_positive_guard(self):
        np.testing.assert_array_equal(positive_weights(split="development_train", positive_counts=[2, 4, 8], sample_count=8),
                                      [3, 1, 0])
        with self.assertRaisesRegex(ValueError, "zero positives"):
            positive_weights(split="development_train", positive_counts=[0, 2], sample_count=8)
        with self.assertRaisesRegex(ValueError, "development-train"):
            positive_weights(split="validation", positive_counts=[2], sample_count=8)

    def test_calibration_rejects_training_and_test(self):
        inputs = {"targets": [[1], [0]], "probabilities": [[0.43], [0.41]]}
        self.assertEqual(fit_validation_threshold(split="validation", **inputs).threshold, 0.43)
        for split in ("development_train", "test", "released_test"):
            with self.assertRaisesRegex(ValueError, "validation"):
                fit_validation_threshold(split=split, **inputs)

    def test_result_round_trip_captures_configuration(self):
        config = {"optimizer": {"name": "adamw"}}
        result = ExperimentResult(config=config, git_commit="abc", timestamp_utc="2026-01-01T00:00:00Z",
                                  seeds={"python": 1, "numpy": 2, "pytorch": 3, "recipe": 4}, device="cpu",
                                  pytorch_version="example", parameter_count=4, train_size=8, validation_size=3,
                                  epoch_history=[{"loss": 0.7}], best_epoch=1, best_validation_loss=0.7,
                                  calibrated_threshold=0.5, final_metrics={"macro_f1": 0.8},
                                  truncation_statistics={"count": 0}, runtime_seconds=0.1)
        config["optimizer"]["name"] = "adam"
        self.assertEqual(result.config["optimizer"]["name"], "adamw")
        self.assertEqual(ExperimentResult.from_bytes(result.to_bytes()), result)

    def test_fingerprint_is_order_independent_but_detects_input_changes(self):
        args = {"source_manifest": b"a", "validation_manifest": b"b", "code_revision": "abc", "seed": 2023}
        a = reproducibility_fingerprint({"a": 1, "b": 2}, **args)
        self.assertEqual(a, reproducibility_fingerprint({"b": 2, "a": 1}, **args))
        self.assertNotEqual(a, reproducibility_fingerprint({"a": 1, "b": 2}, **{**args, "seed": 2024}))
        self.assertNotEqual(a, reproducibility_fingerprint({"a": 1, "b": 2}, **{**args, "validation_manifest": b"changed"}))


class CapacityTests(unittest.TestCase):
    def test_frozen_artificial_config_loads_and_rejects_invalid_values(self):
        config = json.loads(CONFIG.read_bytes())
        validate_config(config)
        for key, value in (("dataset_access", True), ("seed", False), ("sequence_length", 5000),
                           ("candidate_batch_sizes", [4, 8, 32]), ("device", "cpu")):
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_config({**config, key: value})

    def test_small_cpu_capacity_step_has_finite_gradients_and_logits(self):
        result = probe_step(2, 35, device=torch.device("cpu"), seed=2023)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["output_shape"], [2, 4])
        self.assertTrue(result["finite_loss"] and result["finite_gradients"])
        self.assertEqual(result["memory_snapshots"]["after_step"], {})

    def test_unavailable_mps_never_runs_the_probe_or_selects_a_batch(self):
        with patch("scripts.probe_sequence_cnn_capacity.resolve_device", side_effect=RuntimeError("unavailable")):
            with patch("scripts.probe_sequence_cnn_capacity.probe_step") as step:
                result = run_probe(json.loads(CONFIG.read_bytes()))
        step.assert_not_called()
        self.assertIsNone(result["batch_size_recommendation"])
        self.assertEqual([row["status"] for row in result["results"]], ["skipped"] * 3)

    def test_selection_steps_down_from_largest_success(self):
        with patch("scripts.probe_sequence_cnn_capacity.resolve_device", return_value=torch.device("mps")):
            with patch("torch.mps.empty_cache"), patch("scripts.probe_sequence_cnn_capacity.probe_step",
                                                      side_effect=[{"status": "passed", "batch_size": size} for size in (4, 8, 16)]):
                result = run_probe(json.loads(CONFIG.read_bytes()))
        self.assertEqual(result["batch_size_recommendation"], 8)
