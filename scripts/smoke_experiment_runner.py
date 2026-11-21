"""Run one small generic experiment on generated numerical sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_mtwf.cnn import CNNConfig, VariableLengthCNN
from adaptive_mtwf.experiment_runner import DataSource, RunMetadata, RunnerConfig, run_seeds
from adaptive_mtwf.sequence_training import TrainingConfig
from scripts.smoke_sequence_cnn import artificial_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    train = DataSource("development_train", "artificial-sequences-v1:9:16", artificial_sequences(9, 16))
    validation = DataSource("validation", "artificial-sequences-v1:10:8", artificial_sequences(10, 8))
    config = RunnerConfig(4, TrainingConfig(epochs=3, batch_size=8, optimizer="adamw", device=args.device))
    run_seeds(seeds=(2023,), model_factory=lambda: VariableLengthCNN(CNNConfig(4)), train=train, validation=validation,
              config=config, root=args.output, metadata=RunMetadata("generic_runner_smoke"))
    report = json.loads((args.output / "seed_2023" / "metadata.json").read_bytes())
    print(json.dumps({"device": report["device"], "best_epoch": report["best_epoch"],
                      "calibrated_threshold": report["calibrated_threshold"], "runtime_seconds": report["runtime_seconds"],
                      "losses": [{key: row[key] for key in ("epoch", "train_loss", "validation_loss")}
                                 for row in report["epoch_history"]]}, indent=2))


if __name__ == "__main__":
    main()
