"""Tests for gallery standardization, DTW, and nearest neighbors."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vectorchain import FeatureStandardizer, Neighbor, dtw_distance, nearest_neighbors


def test_standardizer_fits_gallery_columns_and_preserves_constant_features() -> None:
    gallery = (
        np.array([[0.0, 5.0], [2.0, 5.0]]),
        np.array([[4.0, 5.0]]),
    )

    standardizer = FeatureStandardizer.fit(gallery)
    transformed = standardizer.transform([[2.0, 5.0]])

    np.testing.assert_allclose(standardizer.mean_, [2.0, 5.0])
    np.testing.assert_allclose(standardizer.scale_, [np.sqrt(8.0 / 3.0), 1.0])
    np.testing.assert_array_equal(transformed, [[0.0, 0.0]])
    assert not standardizer.mean_.flags.writeable
    assert not standardizer.scale_.flags.writeable


def test_standardizer_does_not_refit_on_query_values() -> None:
    standardizer = FeatureStandardizer.fit((np.array([[0.0], [2.0]]),))

    transformed = standardizer.transform([11.0])

    np.testing.assert_array_equal(transformed, [[10.0]])
    np.testing.assert_array_equal(standardizer.mean_, [1.0])


def test_dtw_known_distance_and_diagonal_window() -> None:
    assert dtw_distance([0.0, 1.0], [0.0, 2.0]) == pytest.approx(0.5)
    assert dtw_distance([0.0, 1.0, 2.0], [0.0, 3.0, 2.0], window=0) == pytest.approx(2.0 / 3.0)


def test_dtw_local_cost_is_rms_across_features() -> None:
    left = np.array([[0.0, 0.0]])
    right = np.array([[3.0, 4.0]])

    assert dtw_distance(left, right) == pytest.approx(np.sqrt(12.5))


@pytest.mark.property
@given(
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
    st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
)
def test_dtw_is_symmetric_for_finite_scalar_sequences(
    left: list[float], right: list[float]
) -> None:
    assert dtw_distance(left, right) == pytest.approx(dtw_distance(right, left))


def test_nearest_neighbors_are_ranked_with_stable_ties() -> None:
    references = ([0.0, 0.0], [2.0, 2.0], [2.0, 2.0])

    neighbors = nearest_neighbors([1.8, 2.0], references, k=3)

    assert neighbors[0] == Neighbor(1, pytest.approx(0.1))
    assert neighbors[1] == Neighbor(2, pytest.approx(0.1))
    assert neighbors[2].index == 0


@pytest.mark.parametrize(
    ("left", "right", "error"),
    [
        ([], [1.0], ValueError),
        ([[[1.0]]], [1.0], ValueError),
        (["one"], [1.0], TypeError),
        ([1.0 + 0.0j], [1.0], TypeError),
        ([np.nan], [1.0], ValueError),
        ([[1.0, 2.0]], [[1.0]], ValueError),
    ],
)
def test_dtw_rejects_invalid_sequences(left: object, right: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        dtw_distance(left, right)


@pytest.mark.parametrize(
    ("window", "error"), [(True, TypeError), (1.5, TypeError), (-1, ValueError)]
)
def test_dtw_rejects_invalid_windows(window: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        dtw_distance([0.0], [0.0], window=window)


@pytest.mark.parametrize(
    ("references", "k", "error"),
    [
        ((), 1, ValueError),
        (([0.0],), True, TypeError),
        (([0.0],), 0, ValueError),
        (([0.0],), 2, ValueError),
    ],
)
def test_nearest_neighbors_rejects_invalid_requests(
    references: object, k: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        nearest_neighbors([0.0], references, k=k)


def test_standardizer_rejects_inconsistent_or_invalid_collections() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FeatureStandardizer.fit(())
    with pytest.raises(ValueError, match="same number"):
        FeatureStandardizer.fit((np.ones((2, 1)), np.ones((2, 2))))
    standardizer = FeatureStandardizer.fit((np.ones((2, 1)),))
    with pytest.raises(ValueError, match="exactly 1"):
        standardizer.transform(np.ones((2, 2)))
