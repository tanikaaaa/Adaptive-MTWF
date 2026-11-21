"""Read-only identity and exact-record overlap audit of the released partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from adaptive_mtwf.data.loaders.tmwf import load_single_tab_split
from adaptive_mtwf.data.split_integrity import audit_split_ids

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/data_audits/tmwf_source_integrity.json"


def record_digest(timestamps: NDArray[np.float32], directions: NDArray[np.int8]) -> str:
    """Hash both complete stored channels in order, independently of the label."""
    digest = hashlib.sha256(struct.pack("<QQ", timestamps.size, directions.size))
    digest.update(timestamps.astype("<f4", copy=False).tobytes())
    digest.update(directions.tobytes())
    return digest.hexdigest()


def audit_release(dataset_dir: Path) -> dict:
    source_ids = {}
    record_digests = {}
    splits = {}
    for split in ("train", "test"):
        dataset = load_single_tab_split(dataset_dir, split)
        source_ids[split] = [dataset.trace(index).trace_id for index in range(len(dataset))]
        record_digests[split] = [record_digest(times, directions)
                                 for times, directions in zip(dataset.timestamps, dataset.directions)]
        splits[split] = {
            "source_file": str(dataset.source_file), "source_sha256": dataset.source_sha256,
            "class_counts": dict(sorted(Counter(dataset.labels.tolist()).items())),
        }
        del dataset
    checks = {"source_ids": audit_split_ids(source_ids), "exact_record_contents": audit_split_ids(record_digests)}
    return {
        "scope": "Original released train and test files.",
        "content_identity": "SHA-256 of channel lengths and complete ordered float32 timestamps and int8 directions, excluding labels.",
        "splits": splits,
        "checks": {key: {**asdict(report), "is_disjoint": report.is_disjoint} for key, report in checks.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Trusted local directory containing train and test.")
    args = parser.parse_args()
    report = audit_release(args.dataset_dir)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=ARTIFACT.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(json.dumps(report, indent=2) + "\n")
            file.close()
            temporary.replace(ARTIFACT)
        finally:
            temporary.unlink(missing_ok=True)
    for method, result in report["checks"].items():
        shared = result["overlaps"][0]["shared_id_count"]
        print(f"{method}: {shared} distinct identities shared by the released train and test splits")
    print(f"Audit written to {ARTIFACT}")


if __name__ == "__main__":
    main()
