"""Read the released TMWF Tor Browser single- and multi-tab pickles."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from hashlib import sha256
from operator import index as integer_index
from pathlib import Path
from typing import BinaryIO, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..schema import CanonicalTrace

SOURCE_DATASET = "tmwf-tbb-single-tab"
REAL_SOURCE_DATASET = "tmwf-tbb-real-multitab"
UNMONITORED_LABEL = -1
MONITORED_CLASS_COUNT = 50
REAL_UNMONITORED_LABEL = 50
REAL_COLLECTIONS = {"2tabs": 2, "2tabs-extra": 2, "3tabs": 3, "4tabs": 4}
_Scalar = TypeVar("_Scalar", bound=np.generic)


@dataclass(frozen=True, eq=False)
class TMWFSingleTabSplit:
    """The three source columns, with zero-based sample indices."""

    source_file: Path
    source_sha256: str
    source_size_bytes: int
    split: str
    timestamps: tuple[NDArray[np.float32], ...]
    directions: tuple[NDArray[np.int8], ...]
    labels: NDArray[np.int8]

    def __len__(self) -> int:
        return len(self.labels)

    def trace(self, source_index: int) -> CanonicalTrace:
        """Adapt one complete observation sequence, including any anomalies.

        Widen float32 timestamps to float64 and subtract the first timestamp
        when it is finite. This preserves the source's timing units and cell
        order. Direction signs and the raw label, including -1, are preserved.
        A nonfinite origin leaves the timestamp values available for audit.

        Run ``validate_trace`` to report issues in the returned trace. The
        model conversion and merger enforce validation themselves.
        """
        source_index = integer_index(source_index)
        if not 0 <= source_index < len(self):
            raise IndexError(f"{self.split} source index out of range: {source_index}")
        times = self.timestamps[source_index].astype(np.float64)
        if times.size and np.isfinite(times[0]):
            times -= times[0]
        label = int(self.labels[source_index])
        return CanonicalTrace(
            timestamps=times,
            directions=self.directions[source_index].copy(),
            labels=(label,),
            trace_id=f"{SOURCE_DATASET}:{self.split}:{self.source_sha256}:{source_index}",
            source_dataset=SOURCE_DATASET,
            valid_length=times.size,
            metadata={
                "source_file": str(self.source_file),
                "source_sha256": self.source_sha256,
                "source_index": source_index,
                "split": self.split,
                "original_label": label,
            },
        )


def _read_trace_column(
    column: object, key: str, dtype: type[_Scalar], count: int, path: Path
) -> tuple[NDArray[_Scalar], ...]:
    if not isinstance(column, list) or len(column) != count:
        raise ValueError(f"{path}: {key!r} must be a list of {count} arrays.")
    for index, values in enumerate(column):
        if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != dtype:
            raise ValueError(f"{path}: {key}[{index}] must be a one-dimensional {dtype.__name__} array.")
    return tuple(column)


def _fingerprint(file: BinaryIO) -> tuple[str, int]:
    digest = sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    size = file.tell()
    file.seek(0)
    return digest.hexdigest(), size


def _read_release_pickle(path: Path) -> tuple[dict, str, int]:
    with path.open("rb") as file:
        checksum, size = _fingerprint(file)
        if file.read(2) != b"\x80\x04":
            raise ValueError(f"{path}: expected a protocol-4 pickle.")
        file.seek(0)
        raw = pickle.load(file)
        if file.read(1):
            raise ValueError(f"{path}: unexpected data after the pickle.")
    if not isinstance(raw, dict) or set(raw) != {"time", "data", "label"}:
        raise ValueError(f"{path}: expected the time, data, and label columns.")
    return raw, checksum, size


def load_single_tab_split(dataset_dir: Path, split: str) -> TMWFSingleTabSplit:
    """Open a trusted local TMWF release pickle in read-only mode.

    The directory contains the extensionless release files ``train`` and
    ``test``. Structural schema errors raise with source context. Per-trace
    value and alignment issues remain in the data for inspection.
    """
    if split not in ("train", "test"):
        raise ValueError(f"Unknown TMWF single-tab partition: {split!r}")
    path = (dataset_dir / split).resolve()
    raw, checksum, size = _read_release_pickle(path)
    labels = raw["label"]
    if not isinstance(labels, np.ndarray) or labels.ndim != 1 or labels.dtype != np.int8:
        raise ValueError(f"{path}: 'label' must be a one-dimensional int8 array.")
    if np.any((labels < UNMONITORED_LABEL) | (labels >= MONITORED_CLASS_COUNT)):
        raise ValueError(f"{path}: expected raw labels -1 or 0 through {MONITORED_CLASS_COUNT - 1}.")
    return TMWFSingleTabSplit(
        source_file=path,
        source_sha256=checksum,
        source_size_bytes=size,
        split=split,
        timestamps=_read_trace_column(raw["time"], "time", np.float32, len(labels), path),
        directions=_read_trace_column(raw["data"], "data", np.int8, len(labels), path),
        labels=labels,
    )


@dataclass(frozen=True, eq=False)
class TMWFRealCollection:
    """The original real capture arrays and per-tab label slots for one file."""

    source_file: Path
    source_sha256: str
    source_size_bytes: int
    collection: str
    tab_count: int
    timestamps: tuple[NDArray[np.float32], ...]
    directions: tuple[NDArray[np.int8], ...]
    labels: NDArray[np.int8]

    def __len__(self) -> int:
        return len(self.labels)

    def trace(self, source_index: int) -> CanonicalTrace:
        """Copy one full trace into relative time while preserving raw order.

        Keep every tab label in metadata, even repeated labels and background
        slots. The canonical label tuple is the stable union. Value anomalies
        remain visible to the generic validator.
        """
        source_index = integer_index(source_index)
        if not 0 <= source_index < len(self):
            raise IndexError(f"{self.collection} source index out of range: {source_index}")
        times = self.timestamps[source_index].astype(np.float64)
        origin = float(times[0]) if times.size and np.isfinite(times[0]) else None
        if origin is not None:
            times -= origin
        raw_labels = tuple(self.labels[source_index].tolist())
        return CanonicalTrace(
            timestamps=times,
            directions=self.directions[source_index].copy(),
            labels=tuple(dict.fromkeys(raw_labels)),
            trace_id=f"{REAL_SOURCE_DATASET}:{self.collection}:{self.source_sha256}:{source_index}",
            source_dataset=REAL_SOURCE_DATASET,
            valid_length=times.size,
            metadata={
                "source_file": str(self.source_file), "source_sha256": self.source_sha256,
                "source_index": source_index, "collection": self.collection, "tab_count": self.tab_count,
                "original_tab_labels": raw_labels, "original_timestamp_origin": origin,
            },
        )


def load_real_multitab(dataset_dir: Path, collection: str) -> TMWFRealCollection:
    """Load a trusted local TBB real-world pickle, including rows with anomalies."""
    if collection not in REAL_COLLECTIONS:
        raise ValueError(f"Unknown TMWF real TBB collection: {collection!r}")
    tabs = REAL_COLLECTIONS[collection]
    path = (dataset_dir / collection).resolve()
    raw, checksum, size = _read_release_pickle(path)
    labels = raw["label"]
    if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.shape[1] != tabs or labels.dtype != np.int8:
        raise ValueError(f"{path}: expected an int8 label matrix with {tabs} columns.")
    if np.any((labels < 0) | (labels > REAL_UNMONITORED_LABEL)):
        raise ValueError(f"{path}: expected raw labels 0 through {REAL_UNMONITORED_LABEL}.")
    return TMWFRealCollection(
        source_file=path,
        source_sha256=checksum,
        source_size_bytes=size,
        collection=collection,
        tab_count=tabs,
        timestamps=_read_trace_column(raw["time"], "time", np.float32, len(labels), path),
        directions=_read_trace_column(raw["data"], "data", np.int8, len(labels), path),
        labels=labels,
    )
