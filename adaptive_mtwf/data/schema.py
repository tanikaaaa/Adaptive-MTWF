"""Representations shared by recorded and synthetic Tor cell traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

MAX_CELLS = 5_000  # Legacy fixed representation.
MAX_SOURCE_CELLS = 5_000
MERGE_PROVENANCE_KEY = "merge_provenance"
WebsiteLabel = Union[int, str]


@dataclass(frozen=True, eq=False)
class CanonicalTrace:
    """Store timestamp and direction channels in observed cell order.

    Timestamps are relative to the first valid cell, with units inherited from
    the input. Equal timestamps are allowed. Directions are -1 or +1 in valid
    positions. Labels are distinct integers or nonblank strings, in target
    order. The source dataset and trace ID together identify a source sample.

    With no mask, every cell is valid and the trace can be any positive length.
    Fixed-length traces have three arrays of shape ``(MAX_CELLS,)``. The mask is
    True for the first ``valid_length`` cells. Padded timestamps and directions
    are zero. Use ``make_trace`` to validate and normalize source observations.
    """

    timestamps: NDArray[np.generic]
    directions: NDArray[np.generic]
    labels: tuple[WebsiteLabel, ...]
    trace_id: str
    source_dataset: str
    valid_length: int
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: Optional[NDArray[np.bool_]] = None
