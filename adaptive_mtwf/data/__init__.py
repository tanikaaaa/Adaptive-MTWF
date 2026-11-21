"""Public API for the adaptive MTWF data foundation."""

from .canonicalize import canonicalize_trace, cap_single_tab_trace, inter_arrival_times, make_trace
from .merge import TiePolicy, merge_variable_length, merge_with_offsets
from .schema import MAX_CELLS, MAX_SOURCE_CELLS, CanonicalTrace, WebsiteLabel
from .synthesis import SynthesisRejected, assess_synthesis_eligibility, synthesize_delay, synthesize_ratio
from .validate import TraceValidationError, ValidationIssue, ValidationResult, validate_trace

__all__ = [
    "MAX_CELLS",
    "MAX_SOURCE_CELLS",
    "CanonicalTrace",
    "SynthesisRejected",
    "TiePolicy",
    "TraceValidationError",
    "ValidationIssue",
    "ValidationResult",
    "WebsiteLabel",
    "assess_synthesis_eligibility",
    "canonicalize_trace",
    "cap_single_tab_trace",
    "inter_arrival_times",
    "make_trace",
    "merge_variable_length",
    "merge_with_offsets",
    "synthesize_delay",
    "synthesize_ratio",
    "validate_trace",
]
