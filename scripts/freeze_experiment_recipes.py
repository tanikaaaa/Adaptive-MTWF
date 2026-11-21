"""Freeze development partitions and generic evaluation recipe schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from adaptive_mtwf.data.loaders.tmwf import SOURCE_DATASET, load_single_tab_split
from adaptive_mtwf.data.recipes import RecipeSampler, RecipeSchedule, SourceRecord, json_bytes
from adaptive_mtwf.data.split_integrity import audit_split_ids

ROOT = Path(__file__).resolve().parents[1]
PRIOR_AUDIT = ROOT / "artifacts/data_audits/tmwf_source_integrity.json"
VOCABULARY = tuple(range(50))
GROUP_SIZES = (2, 3, 4, 5)
CONDITIONS = (0.1, 0.2, 0.3, 0.4, 0.5)


def stratify_rows(labels, vocabulary, seed):
    """Allocate floor(n/5) validation rows per class, with at least one per side."""
    partitions = {"development_train": [], "validation": [], "background": []}
    by_label = defaultdict(list)
    for index, value in enumerate(labels):
        by_label[int(value)].append(index)
    partitions["background"] = by_label.pop(-1, [])
    if set(by_label) != set(vocabulary):
        raise ValueError("Released train labels do not match the monitored vocabulary.")
    for label in vocabulary:
        rows = by_label[label]
        if len(rows) < 2:
            raise ValueError(f"Class {label} needs at least two rows for a development split.")
        rng = random.Random(f"development-split-v1:{seed}:{label}")
        rng.shuffle(rows)
        cutoff = max(1, len(rows) // 5)
        partitions["validation"].extend(rows[:cutoff])
        partitions["development_train"].extend(rows[cutoff:])
    return {split: sorted(rows) for split, rows in partitions.items()}


def _source_id(released_split, checksum, row):
    return f"{SOURCE_DATASET}:{released_split}:{checksum}:{row}"


def _schedule(split, seed):
    return RecipeSchedule(split, seed, GROUP_SIZES, CONDITIONS)


def build_artifacts(dataset_dir, seed, evaluation_count, audit_count, prior):
    train = load_single_tab_split(dataset_dir, "train")
    prior_test = prior["splits"]["test"]
    if train.source_sha256 != prior["splits"]["train"]["source_sha256"] or not all(
        prior["checks"][kind]["is_disjoint"] for kind in ("source_ids", "exact_record_contents")
    ):
        raise ValueError("The released train file and the successful prior overlap audit must agree.")
    subsets = stratify_rows(train.labels, VOCABULARY, seed)
    records = {}
    artifacts = {}
    for split, rows in subsets.items():
        records[split] = [SourceRecord(_source_id("train", train.source_sha256, row), int(train.labels[row]), split)
                          for row in rows]
        artifacts[f"{split}_sources.jsonl"] = b"".join(json_bytes({
            **asdict(record), "original_row_index": row, "original_released_split": "train",
            "provenance": {"dataset": SOURCE_DATASET, "source_file": str(train.source_file),
                           "source_sha256": train.source_sha256},
        }) for row, record in zip(rows, records[split]))

    # Reconstruct the complete reserved ID namespace from the prior audit.
    # These IDs are used only for a set intersection, never for recipe draws.
    test_count = prior["checks"]["source_ids"]["row_counts"]["test"]
    partitions = {split: [record.source_id for record in group] for split, group in records.items()}
    partitions["released_test"] = [_source_id("test", prior_test["source_sha256"], row) for row in range(test_count)]
    leakage = audit_split_ids(partitions)
    if not leakage.is_disjoint or any(leakage.row_counts[key] != leakage.unique_id_counts[key] for key in partitions):
        raise ValueError("Source IDs overlap or repeat.")

    def sampler(split):
        return RecipeSampler(_schedule(split, seed), records[split], VOCABULARY, distinct_labels=True)

    validation = sampler("validation")
    test_schedule = _schedule("released_test", seed)
    fixed = {
        "validation_recipes.jsonl": validation.recipe,
        "released_test_schedule.jsonl": test_schedule.recipe,
    }
    validation_ids = set(partitions["validation"])
    for name, recipe_at in fixed.items():
        rows = [recipe_at(index) for index in range(evaluation_count)]
        if name == "validation_recipes.jsonl" and any(not set(row.source_ids) <= validation_ids for row in rows):
            raise AssertionError("Validation recipe used a source from another split.")
        content = b"".join(row.to_bytes() for row in rows)
        if content != b"".join(recipe_at(index).to_bytes() for index in range(evaluation_count)):
            raise AssertionError("Evaluation recipe replay differs.")
        artifacts[name] = content

    train_sampler = sampler("development_train")
    train_labels = {record.source_id: record.label for record in records["development_train"]}
    sizes, conditions, classes = Counter(), Counter(), Counter()
    for index in range(audit_count):
        recipe = train_sampler.recipe(index)
        if (len(set(recipe.source_labels)) != recipe.group_size
                or any(train_labels.get(key) != label for key, label in zip(recipe.source_ids, recipe.source_labels))):
            raise AssertionError("Recipe contains duplicate or invalid labels or sources.")
        sizes[recipe.group_size] += 1
        conditions[recipe.condition] += 1
        classes.update(recipe.source_labels)
    deviations = {
        "group_size_max_absolute_frequency_deviation": max(abs(sizes[size] / audit_count - 1 / len(GROUP_SIZES)) for size in GROUP_SIZES),
        "condition_max_absolute_frequency_deviation": max(abs(conditions[value] / audit_count - 1 / len(CONDITIONS)) for value in CONDITIONS),
        "class_max_relative_participation_deviation": max(abs(classes[label] / (sum(classes.values()) / len(VOCABULARY)) - 1) for label in VOCABULARY),
    }
    tolerances = dict(zip(deviations, (0.03, 0.03, 0.25)))
    report = {
        "seed": seed, "development_validation_fraction": 0.2,
        "recipe_configuration": {"group_sizes": GROUP_SIZES, "conditions": CONDITIONS, "distinct_labels": True,
                                 "python_version": platform.python_version()},
        "source_counts": leakage.row_counts, "source_id_disjoint": leakage.is_disjoint,
        "source_id_overlap": [asdict(item) for item in leakage.overlaps],
        "class_counts": {split: dict(sorted(Counter(record.label for record in group).items())) for split, group in records.items()},
        "released_test": {"opened": False, "sampled": False, "binding": "deferred", "prior_source_sha256": prior_test["source_sha256"]},
        "sample_count": audit_count, "group_size_counts": dict(sorted(sizes.items())),
        "condition_counts": dict(sorted(conditions.items())), "class_participation": dict(sorted(classes.items())),
        "wrong_split_or_unknown_labels_or_duplicate_labels": 0,
        "distribution_deviations": deviations,
        "distribution_audit_tolerances": tolerances,
        "distribution_within_tolerance": all(value <= tolerances[key] for key, value in deviations.items()),
        "evaluation_count_per_split": evaluation_count, "fixed_evaluation_replay_identical": True,
        "manifest_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
    }
    return artifacts, report


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--evaluation-count", type=int, default=256)
    parser.add_argument("--audit-count", type=int, default=5000)
    args = parser.parse_args()
    if args.seed < 0 or min(args.evaluation_count, args.audit_count) < 1:
        parser.error("Use a nonnegative seed and positive sample counts.")
    prior = json.loads(PRIOR_AUDIT.read_bytes())
    artifacts, report = build_artifacts(args.dataset_dir, args.seed, args.evaluation_count, args.audit_count, prior)
    for name, data in artifacts.items():
        _write(ROOT / "artifacts/manifests" / name, data)
    audit = ROOT / "artifacts/data_audits/experiment_recipe_audit.json"
    _write(audit, (json.dumps(report, indent=2) + "\n").encode())
    print(report["source_counts"])
    print(f"Audit written to {audit}")


if __name__ == "__main__":
    main()
