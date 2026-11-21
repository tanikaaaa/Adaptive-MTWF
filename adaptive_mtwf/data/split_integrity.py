"""Aggregate overlap checks for existing partitions of source identities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SplitOverlap:
    left: str
    right: str
    shared_id_count: int


@dataclass(frozen=True)
class SplitIntegrity:
    row_counts: dict[str, int]
    unique_id_counts: dict[str, int]
    overlaps: tuple[SplitOverlap, ...]

    @property
    def is_disjoint(self) -> bool:
        """True when no identity occurs in more than one supplied partition."""
        return all(overlap.shared_id_count == 0 for overlap in self.overlaps)


def audit_split_ids(partitions: Mapping[str, Iterable[str]]) -> SplitIntegrity:
    """Count rows, distinct identities, and pairwise overlap without returning IDs."""
    if len(partitions) < 2:
        raise ValueError("At least two partitions are required for a disjointness check.")
    rows: dict[str, int] = {}
    identities: dict[str, set[str]] = {}
    for split in sorted(partitions):
        rows[split] = 0
        identities[split] = set()
        for identity in partitions[split]:
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"Partition {split!r} contains a missing or invalid identity.")
            identities[split].add(identity)
            rows[split] += 1
    overlaps = tuple(
        SplitOverlap(left, right, len(identities[left] & identities[right]))
        for left, right in combinations(identities, 2)
    )
    return SplitIntegrity(rows, {split: len(ids) for split, ids in identities.items()}, overlaps)
