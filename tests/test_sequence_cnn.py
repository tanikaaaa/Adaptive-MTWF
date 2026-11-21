"""Verify the generic CNN with artificial numerical arrays only."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from adaptive_mtwf.cnn import CNNConfig, VariableLengthCNN, make_bce_loss
from adaptive_mtwf.sequence_training import TrainingConfig, load_checkpoint, resolve_device, save_checkpoint, train_sequences
from scripts.smoke_sequence_cnn import artificial_sequences, run_smoke


class SequenceCNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.config = CNNConfig(num_classes=3, in_channels=4, widths=(6, 8, 10, 12), hidden_size=7)
        self.model = VariableLengthCNN(self.config)

    def test_forward_and_backward_are_finite(self):
        inputs = torch.randn(2, 4, 35, requires_grad=True)
        mask = torch.arange(35)[None, :] < torch.tensor([[19], [33]])
        output = self.model(inputs, mask)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(output.isfinite().all())
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in self.model.parameters()))
        self.assertTrue(inputs.grad.isfinite().all())
        self.assertEqual(inputs.grad.masked_select(~mask[:, None, :]).count_nonzero().item(), 0)

    def test_extra_padding_values_and_other_batch_members_do_not_change_output(self):
        self.model.eval()
        for length in (1, 2, 7, 16, 17, 31, 32, 33):
            with self.subTest(length=length), torch.inference_mode():
                signal = torch.randn(1, 4, length)
                expected = self.model(signal, torch.ones(1, length, dtype=torch.bool))
                for pad in (1, 19):
                    padded = F.pad(signal, (0, pad), value=float("nan"))
                    padded = torch.cat((padded, torch.randn_like(padded)), dim=0)
                    mask = torch.arange(length + pad)[None, :] < torch.tensor([[length], [length + pad]])
                    torch.testing.assert_close(self.model(padded, mask)[:1], expected, atol=2e-6, rtol=2e-6)

    def test_downsampled_masks_round_up_each_block_and_zero_invalid_features(self):
        network = VariableLengthCNN(CNNConfig(3, in_channels=4, widths=(6,), kernels=(3,))).eval()
        inputs = torch.randn(3, 4, 9)
        mask = torch.arange(9)[None, :] < torch.tensor([[1], [4], [9]])
        features, first_mask = network.encode(inputs, mask)
        torch.testing.assert_close(first_mask.sum(1), torch.tensor([1, 2, 5]))
        self.assertEqual(first_mask.shape, (3, 5))
        self.assertEqual(features.masked_select(~first_mask[:, None, :]).count_nonzero().item(), 0)

        long_mask = torch.arange(35)[None, :] < torch.tensor([[1], [16], [17], [35]])
        features, final_mask = self.model.encode(torch.randn(4, 4, 35), long_mask)
        torch.testing.assert_close(final_mask.sum(1), torch.tensor([1, 1, 2, 3]))
        self.assertEqual(features.shape, (4, 12, 3))
        self.assertEqual(features.masked_select(~final_mask[:, None, :]).count_nonzero().item(), 0)

    def test_invalid_masks_and_valid_nonfinite_values_are_rejected(self):
        inputs = torch.ones(1, 4, 3)
        with self.assertRaisesRegex(ValueError, "All-padding"):
            self.model(inputs, torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.model(inputs, torch.tensor([[True, False, True]]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.model(inputs, torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(inputs * torch.inf, torch.ones(1, 3, dtype=torch.bool))

    def test_weighted_and_unweighted_loss_match_known_values(self):
        logits, targets = torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 1.0]])
        base = make_bce_loss(3)(logits, targets)
        weighted = make_bce_loss(3, pos_weight=[2, 1, 3])(logits, targets)
        torch.testing.assert_close(base, torch.log(torch.tensor(2.0)))
        torch.testing.assert_close(weighted, 2 * base)
        for bad in ([1, 1], [-1, 1, 1], [1, float("inf"), 1]):
            with self.assertRaises(ValueError):
                make_bce_loss(3, pos_weight=bad)

    def test_checkpoint_exact_inference(self):
        self.model.eval()
        inputs = torch.randn(2, 4, 23)
        mask = torch.arange(23)[None, :] < torch.tensor([[17], [23]])
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            save_checkpoint(self.model, checkpoint)
            reloaded = load_checkpoint(checkpoint)
        self.assertEqual(reloaded.config, self.model.config)
        with torch.inference_mode():
            torch.testing.assert_close(self.model(inputs, mask), reloaded(inputs, mask), atol=0, rtol=0)


class TrainerTests(unittest.TestCase):
    def test_cpu_training_is_reproducible_and_uses_generic_metrics(self):
        model_config = CNNConfig(4, widths=(4, 6), kernels=(3, 3), hidden_size=5)
        config = TrainingConfig(epochs=1, batch_size=2, optimizer="adam")
        train, validation = artificial_sequences(5, 4), artificial_sequences(6, 4)
        first, history = train_sequences(model_config, train, validation, config, pos_weight=[1, 2, 1, 1])
        replay, repeated_history = train_sequences(model_config, train, validation, config, pos_weight=[1, 2, 1, 1])
        self.assertEqual(history, repeated_history)
        self.assertIn("a_at_true_k", history[0]["validation_metrics"])
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, replay.state_dict()[key], atol=0, rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable in this process")
    def test_mps_forward_backward_and_checkpoint_smoke(self):
        with TemporaryDirectory() as directory:
            result = run_smoke("mps", 1, "adamw", Path(directory) / "mps.pt")
        self.assertEqual(result["device"], "mps")
        self.assertTrue(result["checkpoint_replay_exact"])

    @unittest.skipIf(torch.backends.mps.is_available(), "MPS is available")
    def test_unavailable_mps_request_fails_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
            resolve_device("mps")
