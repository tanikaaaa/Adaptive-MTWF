"""Deterministic recipes over caller-supplied source identifiers and labels."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Sequence, Union

from numpy.typing import ArrayLike

from .batching import SequenceItem, encode_multihot

Label = Union[int, str]
Condition = Union[int, float, str]


def json_bytes(value: object) -> bytes:
    """Use a canonical JSON encoding for persisted recipes and seed derivation."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    label: Label
    split: str


@dataclass(frozen=True)
class SampleRecipe:
    sample_index: int
    round_index: int
    seed: int
    split: str
    group_size: int
    condition: Condition
    # The positions in both tuples define the source order. Empty tuples
    # denote a frozen schedule awaiting a source pool.
    source_ids: tuple[str, ...] = ()
    source_labels: tuple[Label, ...] = ()

    def to_bytes(self) -> bytes:
        return json_bytes(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> SampleRecipe:
        fields = json.loads(data)
        fields["source_ids"] = tuple(fields["source_ids"])
        fields["source_labels"] = tuple(fields["source_labels"])
        recipe = cls(**fields)
        count = len(recipe.source_ids)
        if count != len(recipe.source_labels) or count not in (0, recipe.group_size) or len(set(recipe.source_ids)) != count:
            raise ValueError("Recipe source IDs and labels must describe the same ordered group.")
        return recipe

    @property
    def sample_id(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def sequence_item(self, channels: ArrayLike, vocabulary: Sequence[Label]) -> SequenceItem:
        """Pair a bound recipe with numerical channels provided by the caller."""
        if not self.source_ids:
            raise ValueError("Bind the recipe to sources before attaching channels.")
        return SequenceItem(self.sample_id, channels, encode_multihot(self.source_labels, vocabulary))


@dataclass(frozen=True)
class RecipeSchedule:
    split: str
    base_seed: int
    group_sizes: tuple[int, ...]
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if not self.split or type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("A split and a nonnegative integer base seed are required.")
        sizes, conditions = tuple(self.group_sizes), tuple(self.conditions)
        if not sizes or any(type(size) is not int or size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
            raise ValueError("Group sizes must be unique positive integers.")
        if (not conditions or any(type(value) not in (str, int, float) or
                                  (type(value) is float and not math.isfinite(value)) for value in conditions)
                or len(set(conditions)) != len(conditions)):
            raise ValueError("Conditions must be unique finite numbers or strings.")
        object.__setattr__(self, "group_sizes", sizes)
        object.__setattr__(self, "conditions", conditions)

    def recipe(self, sample_index: int, round_index: int = 0) -> SampleRecipe:
        if any(type(value) is not int or value < 0 for value in (sample_index, round_index)):
            raise ValueError("Sample and round indices must be nonnegative integers.")
        key = json_bytes(["recipe-v1", self.split, self.base_seed, round_index, sample_index])
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = random.Random(seed)
        return SampleRecipe(sample_index, round_index, seed, self.split,
                            rng.choice(self.group_sizes), rng.choice(self.conditions))


class RecipeSampler:
    """Bind a schedule to a pool. Choose labels uniformly in distinct mode."""

    def __init__(self, schedule: RecipeSchedule, sources: Sequence[SourceRecord],
                 vocabulary: Sequence[Label], *, distinct_labels: bool = False) -> None:
        self.schedule = schedule
        self.distinct_labels = distinct_labels
        self.sources = tuple(sorted(sources, key=lambda record: record.source_id))
        vocabulary = tuple(vocabulary)
        if not vocabulary or len(set(vocabulary)) != len(vocabulary) or any(type(x) not in (int, str) for x in vocabulary):
            raise ValueError("Provide a distinct integer/string label vocabulary.")
        groups: dict[Label, list[SourceRecord]] = defaultdict(list)
        for record in self.sources:
            if record.split != schedule.split:
                raise ValueError(f"Source {record.source_id!r} is from the wrong split.")
            if type(record.label) not in (int, str) or record.label not in vocabulary:
                raise ValueError(f"Source {record.source_id!r} has an unknown label.")
            if not record.source_id:
                raise ValueError("Source IDs must be nonempty.")
            groups[record.label].append(record)
        if len({record.source_id for record in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique in a pool.")
        self.groups = tuple(tuple(groups[label]) for label in vocabulary if groups.get(label))
        capacity = len(self.groups) if distinct_labels else len(self.sources)
        if max(schedule.group_sizes) > capacity:
            raise ValueError("The source pool cannot satisfy all group sizes.")

    def bind(self, recipe: SampleRecipe) -> SampleRecipe:
        if recipe != self.schedule.recipe(recipe.sample_index, recipe.round_index):
            raise ValueError("The recipe is already bound or does not belong to this schedule.")
        rng = random.Random(f"sources-v1:{recipe.seed}")
        if self.distinct_labels:
            selected = [rng.choice(group) for group in rng.sample(self.groups, recipe.group_size)]
        else:
            selected = rng.sample(self.sources, recipe.group_size)
        return replace(recipe, source_ids=tuple(record.source_id for record in selected),
                       source_labels=tuple(record.label for record in selected))

    def recipe(self, sample_index: int, round_index: int = 0) -> SampleRecipe:
        return self.bind(self.schedule.recipe(sample_index, round_index))
