"""Descriptive statistics shared by real and synthetic multi-tab audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import CanonicalTrace

TIME_BINS = 20
CANDIDATE_LENGTHS = (5000, 8000, 10000, 12000, 15000, 20000)
FEATURE_DOMAINS = {
    "size_and_time": ("cells", "duration_s", "cells_per_s"),
    "direction": ("outgoing_fraction", "direction_change_rate"),
    "timing": ("iat_mean_s", "iat_p95_s", "iat_p99_s", "zero_iat_fraction", "positive_iat_median_s"),
    "bursts": ("burst_count", "burst_mean_cells", "burst_p95_cells", "burst_max_cells"),
}


@dataclass(frozen=True)
class TraceFeatures:
    scalars: dict[str, float]
    time_profile: NDArray[np.float64]


def trace_features(trace: CanonicalTrace, bins: int = TIME_BINS) -> TraceFeatures:
    """Measure one nonempty, aligned trace with finite times and signed directions.

    Duration is max(time) - min(time), which equals last - first on a
    monotonic trace. All adjacent rates use n-1 pairs. IAT is the signed
    difference in observed order and has no synthetic leading zero. Tied
    adjacent timestamps and zero IAT are equivalent. The signed definition
    allows an explicit raw sensitivity analysis of backward steps.

    Bursts are maximal contiguous equal-direction runs, including at ties.
    Time bins receive equal weight per cell, with the endpoint in the last
    bin. A zero-duration trace goes in bin zero and has undefined cell rate.
    Adjacent statistics are undefined for singleton traces.
    """
    count = trace.valid_length
    if count < 1 or bins < 1:
        raise ValueError("Feature extraction requires a nonempty trace and at least one bin.")
    time = trace.timestamps[:count].astype(np.float64, copy=False)
    direction = trace.directions[:count]
    origin = time.min()
    duration = float(time.max() - origin)
    ia = np.diff(time)
    changes = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    bursts = np.diff(np.r_[0, changes, count])
    burst_median, burst_p90, burst_p95 = np.percentile(bursts, [50, 90, 95]).tolist()
    iat_median, iat_p90, iat_p95, iat_p99 = (
        np.percentile(ia, [50, 90, 95, 99]).tolist() if ia.size else [np.nan] * 4
    )
    positive = ia[ia > 0]
    positive_median, positive_p95 = np.percentile(positive, [50, 95]).tolist() if positive.size else (np.nan, np.nan)
    occupancy = np.zeros(bins, dtype=np.float64)
    if duration:
        positions = np.minimum(((time - origin) / duration * bins).astype(np.intp), bins - 1)
        occupancy = np.bincount(positions, minlength=bins).astype(np.float64) / count
    else:
        occupancy[0] = 1.0
    zero_fraction = np.count_nonzero(ia == 0) / ia.size if ia.size else np.nan
    return TraceFeatures({
        "cells": count, "duration_s": duration, "cells_per_s": count / duration if duration else np.nan,
        "outgoing_fraction": np.count_nonzero(direction == 1) / count,
        "incoming_fraction": np.count_nonzero(direction == -1) / count,
        "direction_change_rate": changes.size / ia.size if ia.size else np.nan,
        "iat_median_s": iat_median, "iat_mean_s": float(ia.mean()) if ia.size else np.nan,
        "iat_p90_s": iat_p90, "iat_p95_s": iat_p95, "iat_p99_s": iat_p99,
        "zero_iat_fraction": zero_fraction, "tied_adjacent_fraction": zero_fraction,
        "positive_iat_median_s": positive_median, "positive_iat_p95_s": positive_p95,
        "burst_count": bursts.size, "burst_mean_cells": float(bursts.mean()),
        "burst_median_cells": burst_median, "burst_max_cells": int(bursts.max()),
        "burst_p90_cells": burst_p90, "burst_p95_cells": burst_p95,
    }, occupancy)


def summarize(values: ArrayLike) -> dict:
    """Describe finite observations with linear empirical quantiles."""
    values = np.asarray(values, dtype=np.float64)
    total = values.size
    values = values[np.isfinite(values)]
    if not values.size:
        return {"defined": 0, "undefined": int(total)}
    names = ("min", "p10", "p25", "median", "p75", "p90", "p95", "p97_5", "p99", "max")
    result = dict(zip(names, np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 97.5, 99, 100]).tolist()))
    return {**result, "mean": float(values.mean()), "iqr": result["p75"] - result["p25"],
            "defined": int(values.size), "undefined": int(total - values.size)}


def summarize_features(rows: Sequence[TraceFeatures]) -> dict:
    if not rows:
        return {"trace_count": 0}
    profiles = np.array([row.time_profile for row in rows])
    return {
        "trace_count": len(rows),
        "scalar_features": {key: summarize([row.scalars[key] for row in rows]) for key in rows[0].scalars},
        "normalized_time_profile": {
            "mean_cell_fraction_by_bin": profiles.mean(axis=0).tolist(),
            "cell_fraction_summary_by_bin": [summarize(column) for column in profiles.T],
        },
    }


def ks_statistic(left: ArrayLike, right: ArrayLike) -> Optional[float]:
    """Compute the maximum empirical CDF difference, including at ties."""
    a, b = np.asarray(left), np.asarray(right)
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not a.size or not b.size:
        return None
    points = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, points, side="right") / a.size
                               - np.searchsorted(b, points, side="right") / b.size)))


def js_divergence(left: ArrayLike, right: ArrayLike) -> float:
    """Base-2 Jensen–Shannon divergence of two normalized profiles in [0, 1]."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a, b = a / a.sum(), b / b.sum()
    midpoint = (a + b) / 2
    summands = []
    for values in (a, b):
        nonzero = values > 0
        summands.append(np.sum(values[nonzero] * np.log2(values[nonzero] / midpoint[nonzero])))
    return float(np.clip(sum(summands) / 2, 0, 1))


def compare_features(real: Sequence[TraceFeatures], synthetic: Sequence[TraceFeatures], real_summary: dict, synthetic_summary: dict) -> dict:
    """Compare trace-level scalars and equally weighted mean time profiles."""
    distances = {}
    for key in real[0].scalars:
        real_median = real_summary["scalar_features"][key].get("median")
        synthetic_median = synthetic_summary["scalar_features"][key].get("median")
        delta = synthetic_median - real_median if real_median is not None and synthetic_median is not None else None
        distances[key] = {
            "ks": ks_statistic([row.scalars[key] for row in real], [row.scalars[key] for row in synthetic]),
            "median_difference_synthetic_minus_real": delta,
            "absolute_median_difference": abs(delta) if delta is not None else None,
            "relative_median_difference": delta / real_median if delta is not None and real_median else None,
        }
    domains = {name: float(np.mean([distances[metric]["ks"] for metric in metrics]))
               for name, metrics in FEATURE_DOMAINS.items()}
    left = real_summary["normalized_time_profile"]["mean_cell_fraction_by_bin"]
    right = synthetic_summary["normalized_time_profile"]["mean_cell_fraction_by_bin"]
    return {
        "real_count": len(real), "synthetic_count": len(synthetic), "scalars": distances,
        "ks_by_domain": domains, "mean_domain_ks": float(np.mean(list(domains.values()))),
        "mean_time_profile_js_divergence_base2": js_divergence(left, right),
    }


def length_coverage(lengths: ArrayLike) -> dict:
    """Evaluate candidate windows over positive observed lengths, with no padding."""
    lengths = np.asarray(lengths)
    lengths = lengths[lengths > 0]
    if not lengths.size:
        return {"trace_count": 0, "candidates": {}}
    candidates = {}
    for limit in CANDIDATE_LENGTHS:
        truncated = lengths > limit
        retained = np.minimum(lengths, limit) / lengths
        p10, p25 = np.percentile(retained[truncated], [10, 25]).tolist() if truncated.any() else (None, None)
        candidates[str(limit)] = {
            "fully_retained_pct": 100 * float(np.mean(~truncated)),
            "truncated_pct": 100 * float(np.mean(truncated)),
            "median_fraction_retained": float(np.median(retained)),
            "truncated_p10_fraction_retained": p10, "truncated_p25_fraction_retained": p25,
        }
    return {"trace_count": int(lengths.size), "length_cells": summarize(lengths), "candidates": candidates}
