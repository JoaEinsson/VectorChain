"""Tests for fitted piecewise-linear reconstruction."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vectorchain import VectorChain


@pytest.mark.parametrize(
    "features",
    [
        ("dt", "dy"),
        ("dy", "dt"),
        ("theta", "dy", "dt"),
        ("dt", "dy", "theta", "r", "delta_theta", "delta_r"),
    ],
)
def test_inverse_transform_supports_feature_ablations_and_order(
    features: tuple[str, ...],
) -> None:
    signal = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    vc = VectorChain(tolerance=0.0, features=features)

    vectors = vc.fit_transform(signal)
    reconstructed = vc.inverse_transform(vectors)

    np.testing.assert_allclose(reconstructed, signal)
    assert reconstructed.dtype == np.float64
    assert reconstructed.flags.writeable


def test_inverse_transform_uses_supplied_displacements_on_fitted_structure() -> None:
    vc = VectorChain(tolerance=0.0, features=("dy", "dt"))
    vectors = vc.fit_transform([0.0, 1.0, 2.0, 2.0, 2.0])
    modified = vectors.copy()
    modified[:, 0] = [4.0, -2.0]

    reconstructed = vc.inverse_transform(modified)

    np.testing.assert_allclose(reconstructed, [0.0, 2.0, 4.0, 3.0, 2.0])


def test_fit_transform_exposes_reconstruction_and_compression_metrics() -> None:
    signal = np.array([0.0, 1.0, 0.0])
    vc = VectorChain(tolerance=0.0, min_segment_length=3)

    vc.fit_transform(signal)

    np.testing.assert_allclose(vc.inverse_transform(vc.vectors_), [0.0, 0.0, 0.0])
    assert vc.compression_factor_ == 3.0
    assert vc.compression_ratio_ == vc.compression_factor_
    assert vc.retention_fraction_ == pytest.approx(1.0 / 3.0)
    assert vc.reconstruction_error_ == pytest.approx(np.sqrt(1.0 / 3.0))


def test_inverse_transform_requires_a_finalized_chain() -> None:
    vc = VectorChain()

    with pytest.raises(RuntimeError, match="finalized fitted chain"):
        vc.inverse_transform(np.empty((0, len(vc.features))))

    vc.update(0.0)
    vc.update(1.0)
    with pytest.raises(RuntimeError, match="finalized fitted chain"):
        vc.inverse_transform(np.empty((0, len(vc.features))))


@pytest.mark.parametrize(
    ("vectors", "error", "message"),
    [
        ([1.0, 1.0], ValueError, "two-dimensional"),
        ([[1.0]], ValueError, "columns"),
        ([[1.0, 1.0], [1.0, 1.0]], ValueError, "rows"),
        ([[1.0, np.nan]], ValueError, "finite"),
        ([[1.0, 1.0 + 0.0j]], TypeError, "real numeric"),
        ([["1", "1"]], TypeError, "real numeric"),
        ([[2.0, 1.0]], ValueError, "dt column"),
    ],
)
def test_inverse_transform_rejects_invalid_vector_matrices(
    vectors: object, error: type[Exception], message: str
) -> None:
    vc = VectorChain(tolerance=0.0, features=("dt", "dy"))
    vc.fit_transform([0.0, 1.0])

    with pytest.raises(error, match=message):
        vc.inverse_transform(vectors)


@pytest.mark.property
@given(
    st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=80,
    ),
    st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_reconstruction_preserves_shape_endpoint_order_and_hinges(
    signal: list[float], tolerance: float
) -> None:
    observed = np.asarray(signal, dtype=np.float64)
    vc = VectorChain(tolerance=tolerance, features=("dy", "dt"))

    vectors = vc.fit_transform(observed)
    reconstructed = vc.inverse_transform(vectors)

    assert reconstructed.shape == observed.shape
    assert reconstructed[0] == pytest.approx(observed[0])
    assert reconstructed[-1] == pytest.approx(observed[-1])
    for segment in vc.segments_:
        assert reconstructed[segment.start] == pytest.approx(segment.start_value)
        assert reconstructed[segment.end] == pytest.approx(segment.end_value)
