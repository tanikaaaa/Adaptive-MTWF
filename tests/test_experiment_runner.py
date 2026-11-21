"""Integration tests for the generic experiment engine on numerical fixtures."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Subset, TensorDataset

from adaptive_mtwf.cnn import CNNConfig, VariableLengthCNN
from adaptive_mtwf.data.batching import SequenceItem, collate_sequences
from adaptive_mtwf.experiment_metadata import ExperimentResult
from adaptive_mtwf.experiment_runner import (
    DataSource, EarlyStopping, Monitor, RunMetadata, RunnerConfig, aggregate_results,
    default_metrics, restore_checkpoint, run_experiment, run_seeds, select_device,
)
from adaptive_mtwf.sequence_training import TrainingConfig
from scripts.smoke_sequence_cnn import artificial_sequences


def make_model():
    torch.manual_seed(31)
    return VariableLengthCNN(CNNConfig(4, widths=(6, 8), kernels=(3, 3), hidden_size=8))


class TrackingModel(VariableLengthCNN):
    def forward(self, inputs, mask):
        if not self.training:
            assert not torch.is_grad_enabled()
            assert all(parameter.grad is None for parameter in self.parameters())
        return super().forward(inputs, mask)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.train = DataSource("development_train", "artificial-train", artificial_sequences(3, 6), "train-v1")
        self.val = DataSource("validation", "artificial-val", artificial_sequences(4, 4), "val-v1")
        self.config = RunnerConfig(4, TrainingConfig(epochs=2, batch_size=2, device="cpu", optimizer="adam"),
                                   pos_weight=(2, 1, 1, 1))
        self.metadata = RunMetadata("wrapper-test", code_revision="fixture-revision")

    def run_case(self, folder, **kwargs):
        options = {"train": self.train, "validation": self.val, "model": make_model(), "config": self.config,
                   "output": self.root / folder, "metadata": self.metadata, **kwargs}
        return run_experiment(**options)

    def test_training_artifacts_metrics_checkpoint_and_calibration(self):
        config = replace(self.config, training=replace(self.config.training, max_length=24))
        model = TrackingModel(make_model().config)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        report = self.run_case("single", model=model, config=config)
        output = self.root / "single"
        for name in ("latest.pt", "best.pt", "history.json", "history.csv", "metadata.json", "config.json",
                     "fingerprint.json", "threshold.json", "final_validation.json", "runtime.json", "best_checkpoint.json"):
            self.assertTrue((output / name).is_file())
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
        self.assertTrue(all(math.isfinite(row["train_loss"]) for row in report.epoch_history))
        self.assertEqual(report.best_validation_loss, min(row["validation_loss"] for row in report.epoch_history))
        self.assertGreater(report.truncation_statistics["train_by_epoch"][0]["truncated"], 0)
        self.assertIn("micro_roc_auc", report.final_metrics["calibrated"])
        self.assertEqual(ExperimentResult.from_bytes((output / "metadata.json").read_bytes()), report)
        self.assertEqual(json.loads((output / "threshold.json").read_bytes())["split"], "validation")
        restored = restore_checkpoint(make_model().eval(), output / "best.pt")
        model2 = make_model().eval()
        model2.load_state_dict(restored["model_state"])
        batch = collate_sequences(self.val.data, max_length=24)
        with torch.inference_mode():
            x, mask = torch.from_numpy(batch.inputs), torch.from_numpy(batch.valid_mask)
            torch.testing.assert_close(model(x, mask), model2(x, mask), rtol=0, atol=0)

    def test_resume_preserves_history_and_matches_uninterrupted_model_and_optimizer(self):
        shorter = replace(self.config, training=replace(self.config.training, epochs=1))
        first = self.run_case("resumed", config=shorter)
        result = self.run_case("resumed", resume=True)
        baseline = self.run_case("straight")
        self.assertEqual(len(result.epoch_history), 2)
        self.assertEqual(result.epoch_history[0], first.epoch_history[0])
        self.assertEqual(result.final_metrics, baseline.final_metrics)
        checkpoints = [torch.load(self.root / name / "latest.pt", weights_only=True) for name in ("resumed", "straight")]
        for key in ("model_state", "optimizer_state"):
            torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Incompatible resume"):
            self.run_case("resumed", resume=True, config=replace(self.config, weight_decay=0.1))
        with self.assertRaisesRegex(ValueError, "Incompatible resume"):
            self.run_case("resumed", resume=True, validation=replace(self.val, content_hash="modified"))

    def test_early_stopping_delta_and_tie_checkpoint(self):
        stop = EarlyStopping(Monitor("macro_f1", "max"), patience=2, min_delta=0.01)
        rows = [{"validation_metrics": {"macro_f1": value}} for value in (0.2, 0.3, 0.305, 0.30)]
        self.assertFalse(stop.stopped(rows[:3]))
        self.assertTrue(stop.stopped(rows))
        config = replace(self.config, training=replace(self.config.training, epochs=4),
                         checkpoint=Monitor("constant", "max"),
                         early_stopping=EarlyStopping(Monitor("constant", "max"), patience=1))
        report = self.run_case("stopped", config=config,
                               metrics=lambda y, p, threshold: {**default_metrics(y, p, threshold), "constant": 0.5})
        self.assertEqual(len(report.epoch_history), 2)
        self.assertEqual(report.best_epoch, 1)

    def test_prebatched_loader_and_holdout_guard(self):
        batch = collate_sequences(self.val.data)
        dataset = TensorDataset(*(torch.from_numpy(array) for array in (batch.inputs, batch.valid_mask, batch.targets)))
        dataset.split = "test"
        declared = replace(self.val, data=DataLoader(Subset(dataset, [0, 1, 2, 3]), batch_size=2))
        with self.assertRaisesRegex(ValueError, "loader split"):
            self.run_case("bad", validation=declared)
        with self.assertRaisesRegex(ValueError, "loader split"):
            self.run_case("bad", validation=replace(self.val, data=Subset(dataset, [0])))
        self.assertFalse((self.root / "bad").exists())
        with self.assertRaisesRegex(ValueError, "Expected the validation split"):
            self.run_case("bad", validation=replace(self.val, split="holdout"))
        dataset.split = "validation"
        report = self.run_case("loader", validation=declared)
        self.assertEqual(report.validation_size, 4)
        self.assertEqual(report.truncation_statistics["validation"]["samples_with_original_lengths"], 0)

    def test_invalid_data_logits_gradients_and_monitor_fail(self):
        bad_row = SequenceItem("invalid", [[1.0], [0.0]], [0, 0, 0.5, 1])
        with self.assertRaisesRegex(ValueError, "binary"):
            self.run_case("bad-target", train=replace(self.train, data=[bad_row]))
        bad_channels = replace(bad_row, channels=[[float("inf")], [0]], target=[0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_case("bad-input", train=replace(self.train, data=[bad_channels]))
        with self.assertRaisesRegex(ValueError, "Unknown validation metric"):
            self.run_case("bad-monitor", config=replace(self.config, checkpoint=Monitor("typo")))
        batch = (torch.zeros(1, 2, 3), torch.tensor([[True, False, True]]), torch.zeros(1, 4))
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            self.run_case("bad-mask", train=replace(self.train, data=DataLoader([batch], batch_size=None)))
        model = make_model()
        hook = model.register_forward_hook(lambda _model, _args, output: output * float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite model logits"):
            self.run_case("bad-output", model=model)
        hook.remove()
        hook = next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "nonfinite gradients"):
            self.run_case("bad-grad", model=model)
        hook.remove()
        self.assertFalse((self.root / "bad-grad" / "latest.pt").exists())

    def test_two_independent_seed_outputs_and_aggregate(self):
        root = self.root / "seeds"
        config = replace(self.config, training=replace(self.config.training, epochs=1))
        factory = lambda: VariableLengthCNN(CNNConfig(4, widths=(4,), kernels=(3,), hidden_size=5))
        options = {"seeds": [11, 12], "model_factory": factory, "train": self.train, "validation": self.val,
                   "config": config, "root": root, "metadata": self.metadata}
        result = run_seeds(**options)
        reports = [ExperimentResult.from_bytes((root / f"seed_{seed}" / "metadata.json").read_bytes()) for seed in (11, 12)]
        self.assertEqual([report.seeds["base"] for report in reports], [11, 12])
        self.assertNotEqual(reports[0].final_metrics["validation_loss"], reports[1].final_metrics["validation_loss"])
        self.assertEqual(result["successful_runs"], 2)
        self.assertEqual(json.loads((root / "aggregate.json").read_bytes()), result)
        known = aggregate_results([replace(reports[0], final_metrics={"fixed": {"score": value}}) for value in (0.25, 0.75)])
        self.assertEqual(known["metrics"]["fixed.score"], {"mean": 0.5, "std": 0.25, "min": 0.25, "max": 0.75, "successful_runs": 2})
        with self.assertRaisesRegex(ValueError, "empty directory"):
            run_seeds(**options)

    def test_auto_device_and_explicit_unavailable_mps(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(select_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
                self.run_case("mps", config=replace(self.config, training=replace(self.config.training, device="mps")))

    def test_fingerprint_replay_with_same_model_and_settings(self):
        config = replace(self.config, training=replace(self.config.training, epochs=1))
        self.run_case("first", config=config)
        self.run_case("second", config=config)
        read = lambda folder: (self.root / folder / "fingerprint.json").read_bytes()
        self.assertEqual(read("first"), read("second"))

    def test_active_output_writer_is_rejected(self):
        output = self.root / "busy"
        output.mkdir()
        with FileLock(str(output / ".experiment.lock")):
            with self.assertRaisesRegex(RuntimeError, "Another writer"):
                self.run_case("busy")
        self.assertFalse((output / "latest.pt").exists())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_trains_and_finalizes(self):
        config = replace(self.config, training=replace(self.config.training, epochs=1, device="mps"))
        self.assertEqual(self.run_case("mps", config=config).device, "mps")
