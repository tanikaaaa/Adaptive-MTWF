"""Test generic recipe reproducibility and the train-only development split."""

import hashlib
import json
import pickle
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from adaptive_mtwf.data.batching import collate_sequences
from adaptive_mtwf.data.recipes import RecipeSampler, RecipeSchedule, SampleRecipe, SourceRecord
from scripts.freeze_experiment_recipes import VOCABULARY, build_artifacts, load_single_tab_split


class GenericRecipeTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = tuple("abcdef")
        self.schedule = RecipeSchedule("dev", 2023, (2, 3, 4, 5), (0.1, 0.2, 0.3, 0.4, 0.5))
        self.sources = [SourceRecord(f"{label}-{i}", label, "dev")
                        for label in self.vocabulary for i in range(3)]
        self.sampler = RecipeSampler(self.schedule, self.sources, self.vocabulary, distinct_labels=True)

    def test_stable_round_trip_and_identity(self):
        a = self.sampler.recipe(8, 3)
        b = RecipeSampler(self.schedule, self.sources[::-1], self.vocabulary, distinct_labels=True).recipe(8, 3)
        self.assertEqual(a.to_bytes(), b.to_bytes())
        self.assertEqual(SampleRecipe.from_bytes(a.to_bytes()), a)
        self.assertNotEqual(a.sample_id, self.sampler.recipe(9, 3).sample_id)
        self.assertNotEqual(a.seed, self.sampler.recipe(8, 4).seed)
        self.assertGreater(len({self.sampler.recipe(i).source_ids for i in range(20)}), 1)

    def test_uniform_draws_and_distinct_labels(self):
        rows = [self.sampler.recipe(index) for index in range(3000)]
        for key, expected in (("group_size", 0.25), ("condition", 0.20)):
            counts = Counter(getattr(row, key) for row in rows)
            self.assertTrue(all(abs(count / len(rows) - expected) < 0.03 for count in counts.values()))
        self.assertEqual({row.group_size for row in rows}, set(self.schedule.group_sizes))
        self.assertEqual({row.condition for row in rows}, set(self.schedule.conditions))
        for row in rows:
            self.assertEqual(len(set(row.source_ids)), row.group_size)
            self.assertEqual(len(set(row.source_labels)), row.group_size)

    def test_reject_wrong_split_and_unknown_label(self):
        for record in (SourceRecord("wrong", "a", "test"), SourceRecord("unknown", "other", "dev")):
            with self.assertRaises(ValueError):
                RecipeSampler(self.schedule, [*self.sources, record], self.vocabulary)

    def test_unbound_schedule_can_be_bound_with_explicit_pool(self):
        unbound = self.schedule.recipe(3)
        restored = SampleRecipe.from_bytes(unbound.to_bytes())
        self.assertFalse(restored.source_ids)
        self.assertEqual(self.sampler.bind(restored), self.sampler.recipe(3))
        with self.assertRaises(ValueError):
            self.sampler.bind(RecipeSchedule("test", 2023, (2,), (0.1,)).recipe(3))

    def test_batching_uses_external_channels_and_recipe_identity(self):
        recipe = self.sampler.recipe(3)
        batch = collate_sequences([recipe.sequence_item([[0, 1, 2], [0, 0, 1]], self.vocabulary)])
        self.assertEqual(batch.sample_ids, (recipe.sample_id,))
        self.assertEqual(batch.inputs.shape, (1, 2, 3))
        self.assertEqual(int(batch.targets.sum()), recipe.group_size)


class DevelopmentSplitTests(unittest.TestCase):
    def test_source_manifests_and_evaluation_replay_from_train_only(self):
        labels = np.array([label for label in VOCABULARY for _ in range(80)] + [-1] * 4, dtype=np.int8)
        times = np.array([0, 1], dtype=np.float32)
        directions = np.array([1, -1], dtype=np.int8)
        raw = pickle.dumps({"label": labels, "time": [times] * len(labels),
                            "data": [directions] * len(labels)}, protocol=4)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "train").write_bytes(raw)
            prior = {"splits": {"train": {"source_sha256": hashlib.sha256(raw).hexdigest()},
                                "test": {"source_sha256": "fixture-held-out"}},
                     "checks": {"source_ids": {"is_disjoint": True, "row_counts": {"test": 10}},
                                "exact_record_contents": {"is_disjoint": True}}}
            with patch("scripts.freeze_experiment_recipes.load_single_tab_split", wraps=load_single_tab_split) as loader:
                files, audit = build_artifacts(path, 2023, 12, 64, prior)
                replay, repeated_audit = build_artifacts(path, 2023, 12, 64, prior)
                self.assertEqual([call.args[1] for call in loader.call_args_list], ["train", "train"])
            self.assertEqual([entry.name for entry in path.iterdir()], ["train"])
        self.assertEqual(files, replay)
        self.assertEqual(audit, repeated_audit)
        self.assertTrue(audit["source_id_disjoint"])
        self.assertEqual(audit["class_counts"]["development_train"], dict.fromkeys(VOCABULARY, 64))
        self.assertEqual(audit["class_counts"]["validation"], dict.fromkeys(VOCABULARY, 16))
        self.assertEqual(audit["class_counts"]["background"], {-1: 4})
        validation_ids = {json.loads(line)["source_id"] for line in files["validation_sources.jsonl"].splitlines()}
        for line in files["validation_recipes.jsonl"].splitlines():
            row = SampleRecipe.from_bytes(line)
            self.assertTrue(set(row.source_ids) <= validation_ids)
        for line in files["released_test_schedule.jsonl"].splitlines():
            row = SampleRecipe.from_bytes(line)
            self.assertEqual(row.split, "released_test")
            self.assertEqual(row.source_ids, ())
            self.assertEqual(row.source_labels, ())
