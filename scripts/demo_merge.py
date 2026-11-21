"""Run from the MTWF directory with ``python3 -m scripts.demo_merge``."""

from adaptive_mtwf.data import CanonicalTrace, make_trace, merge_with_offsets


def cells(trace: CanonicalTrace) -> list[tuple[float, int]]:
    """Return valid (timestamp, direction) pairs for the artificial example."""
    return list(zip(
        trace.timestamps[: trace.valid_length].tolist(),
        trace.directions[: trace.valid_length].tolist(),
    ))


def main() -> None:
    traces = [
        make_trace(
            [0.0, 0.5, 1.0], [1, -1, 1],
            labels=("example-a",), trace_id="demo-a", source_dataset="artificial-demo",
        ),
        make_trace(
            [0.0, 0.25, 0.75], [-1, 1, -1],
            labels=("example-b",), trace_id="demo-b", source_dataset="artificial-demo",
        ),
    ]
    offsets = [0.0, 0.25]
    for trace in traces:
        print(f"Original {trace.trace_id} (relative timestamp, direction): {cells(trace)}")
    print(f"Start offsets: {offsets}")

    mixed = merge_with_offsets(traces, offsets)
    print(f"Chronological merged cells: {cells(mixed)}")
    print(f"Labels: {mixed.labels}")
    print(f"Valid length: {mixed.valid_length}")
    print(
        f"Shapes: timestamps {mixed.timestamps.shape}, directions {mixed.directions.shape}, "
        f"mask {mixed.mask.shape}"
    )


if __name__ == "__main__":
    main()
