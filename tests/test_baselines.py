"""Tests for mandatory sequence baselines."""

import numpy as np
import pytest

from vectorchain import (
    first_difference,
    first_second_difference,
    fixed_linear_segmentation,
    normalized_raw_values,
    raw_values,
)
from vectorchain.features import DEFAULT_FEATURES, compute_features_from_displacements


def test_raw_and_difference_representations_have_explicit_sequence_axes() -> None:
    signal = np.array([1.0, 2.0, 4.0, 7.0])

    np.testing.assert_array_equal(raw_values(signal), [[1.0], [2.0], [4.0], [7.0]])
    np.testing.assert_array_equal(first_difference(signal), [[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(first_second_difference(signal), [[2.0, 1.0], [3.0, 1.0]])
    assert raw_values(signal).dtype == np.float64


def test_normalized_raw_is_per_series_and_handles_constant_inputs() -> None:
    normalized = normalized_raw_values([1.0, 2.0, 3.0])

    assert float(np.mean(normalized)) == pytest.approx(0.0, abs=1e-15)
    assert float(np.std(normalized)) == pytest.approx(1.0)
    np.testing.assert_array_equal(normalized_raw_values([4.0, 4.0, 4.0]), np.zeros((3, 1)))


def test_fixed_segmentation_is_articulated_and_keeps_a_short_tail() -> None:
    result = fixed_linear_segmentation(
        [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
        segment_length=2,
        features=("dy", "dt", "theta"),
    )

    np.testing.assert_array_equal(result.boundaries, [[0, 2], [2, 4], [4, 5]])
    np.testing.assert_allclose(
        result.vectors,
        [
            [2.0, 2.0, np.pi / 4.0],
            [-1.0, 2.0, np.arctan2(-1.0, 2.0)],
            [-1.0, 1.0, -np.pi / 4.0],
        ],
    )
    assert result.feature_names == ("dy", "dt", "theta")
    assert not result.boundaries.flags.writeable
    assert not result.vectors.flags.writeable
    np.testing.assert_array_equal(result.boundaries[1:, 0], result.boundaries[:-1, 1])


def test_fixed_segmentation_uses_the_same_canonical_feature_formulas() -> None:
    signal = np.array([0.0, 2.0, 2.0, 5.0, 4.0])
    fixed = fixed_linear_segmentation(signal, segment_length=2, features=DEFAULT_FEATURES)
    dt = np.array([2.0, 2.0])
    dy = np.array([2.0, 2.0])

    expected = compute_features_from_displacements(dt, dy, DEFAULT_FEATURES)

    np.testing.assert_array_equal(fixed.vectors, expected)


@pytest.mark.parametrize(
    ("function", "series", "error"),
    [
        (raw_values, [], ValueError),
        (raw_values, [[1.0]], ValueError),
        (raw_values, ["one"], TypeError),
        (raw_values, [1.0 + 0.0j], TypeError),
        (raw_values, [np.nan], ValueError),
        (first_difference, [1.0], ValueError),
        (first_second_difference, [1.0, 2.0], ValueError),
    ],
)
def test_baselines_reject_invalid_series(
    function: object, series: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        function(series)


@pytest.mark.parametrize(
    ("segment_length", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_fixed_segmentation_rejects_invalid_durations(
    segment_length: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        fixed_linear_segmentation([0.0, 1.0], segment_length=segment_length)


def test_feature_computation_rejects_invalid_displacements() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_features_from_displacements(np.array([1.0]), np.array([1.0, 2.0]), DEFAULT_FEATURES)
    with pytest.raises(ValueError, match="positive"):
        compute_features_from_displacements(np.array([0.0]), np.array([1.0]), DEFAULT_FEATURES)
    with pytest.raises(ValueError, match="finite"):
        compute_features_from_displacements(np.array([1.0]), np.array([np.nan]), DEFAULT_FEATURES)


def test_empty_displacements_preserve_requested_feature_width() -> None:
    result = compute_features_from_displacements(
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        DEFAULT_FEATURES,
    )

    assert result.shape == (0, len(DEFAULT_FEATURES))
