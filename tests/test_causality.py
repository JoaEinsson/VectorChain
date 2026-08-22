"""Property tests for causal segmentation semantics."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vectorchain import VectorChain

FINITE_VALUE = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
SIGNAL = st.lists(FINITE_VALUE, min_size=2, max_size=30)
TOLERANCE = st.floats(
    min_value=0.0,
    max_value=20.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(signal=SIGNAL, tolerance=TOLERANCE)
def test_batch_wrapper_matches_online_state_machine(signal: list[float], tolerance: float) -> None:
    batch = VectorChain(tolerance=tolerance)
    online = VectorChain(tolerance=tolerance)

    batch_vectors = batch.fit_transform(signal)
    emitted = []
    for value in signal:
        emitted.extend(online.update(value))
    emitted.extend(online.finalize())

    assert tuple(emitted) == online.segments_ == batch.segments_
    np.testing.assert_array_equal(online.segment_boundaries_, batch.segment_boundaries_)
    np.testing.assert_allclose(online.vectors_, batch_vectors, rtol=0.0, atol=0.0)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    prefix=st.lists(FINITE_VALUE, min_size=2, max_size=20),
    left_suffix=st.lists(FINITE_VALUE, max_size=10),
    right_suffix=st.lists(FINITE_VALUE, max_size=10),
    tolerance=TOLERANCE,
)
def test_changing_future_never_mutates_emitted_prefix(
    prefix: list[float],
    left_suffix: list[float],
    right_suffix: list[float],
    tolerance: float,
) -> None:
    left = VectorChain(tolerance=tolerance)
    right = VectorChain(tolerance=tolerance)

    for value in prefix:
        left.update(value)
        right.update(value)

    prefix_segments = left.segments_
    prefix_vectors = left.vectors_.copy()
    assert right.segments_ == prefix_segments
    np.testing.assert_array_equal(right.vectors_, prefix_vectors)

    for value in left_suffix:
        left.update(value)
    for value in right_suffix:
        right.update(value)

    count = len(prefix_segments)
    assert left.segments_[:count] == prefix_segments
    assert right.segments_[:count] == prefix_segments
    np.testing.assert_array_equal(left.vectors_[:count], prefix_vectors)
    np.testing.assert_array_equal(right.vectors_[:count], prefix_vectors)


def test_different_suffixes_preserve_a_known_emitted_segment() -> None:
    left = VectorChain(tolerance=0.0)
    right = VectorChain(tolerance=0.0)

    for value in [0.0, 1.0, 2.0, 2.0]:
        left.update(value)
        right.update(value)

    snapshot = left.segments_
    assert len(snapshot) == 1
    assert snapshot[0].emitted_at == 3

    for value in [2.0, 2.0, 2.0]:
        left.update(value)
    for value in [-100.0, 100.0, -100.0]:
        right.update(value)

    assert left.segments_[0] == right.segments_[0] == snapshot[0]


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(signal=SIGNAL, tolerance=TOLERANCE)
def test_finalized_boundaries_cover_signal_as_an_articulated_chain(
    signal: list[float], tolerance: float
) -> None:
    vc = VectorChain(tolerance=tolerance)

    vc.fit_transform(signal)
    boundaries = vc.segment_boundaries_

    assert boundaries[0, 0] == 0
    assert boundaries[-1, 1] == len(signal) - 1
    assert np.all(boundaries[:, 0] < boundaries[:, 1])
    np.testing.assert_array_equal(boundaries[1:, 0], boundaries[:-1, 1])
    assert all(
        segment.emitted_at is None or segment.emitted_at > segment.end for segment in vc.segments_
    )
