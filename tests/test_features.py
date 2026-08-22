"""Tests for configurable feature projection."""

import numpy as np
import pytest

from vectorchain import DEFAULT_FEATURES, SUPPORTED_FEATURES, VectorChain
from vectorchain.features import compute_feature_matrix, validate_feature_names

ABLATIONS = (
    ("dt", "dy"),
    ("dt", "dy", "theta"),
    ("dt", "dy", "theta", "r"),
    ("dt", "dy", "theta", "r", "delta_theta"),
    ("dt", "dy", "theta", "r", "delta_theta", "delta_r"),
)


@pytest.mark.parametrize(
    ("features", "error"),
    [
        ("dt", TypeError),
        ((), ValueError),
        (("dt", "dy", "dy"), ValueError),
        (("dt", "dy", "unknown"), ValueError),
        (("dt",), ValueError),
        (("dy",), ValueError),
        (("dt", "dy", 1), TypeError),
    ],
)
def test_invalid_feature_selections_are_rejected(features: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        VectorChain(features=features)


def test_feature_order_is_preserved_in_output_columns() -> None:
    vc = VectorChain(tolerance=0.0, features=("dy", "dt"))

    vectors = vc.fit_transform([0.0, 1.0, 2.0])

    assert vc.feature_names_ == ("dy", "dt")
    np.testing.assert_array_equal(vectors, [[2.0, 2.0]])


@pytest.mark.parametrize("features", ABLATIONS)
def test_every_planned_ablation_is_directly_configurable(features: tuple[str, ...]) -> None:
    vc = VectorChain(tolerance=0.0, features=features)

    vectors = vc.fit_transform([0.0, 1.0, 2.0])

    assert vc.feature_names_ == features
    assert vectors.shape == (1, len(features))


def test_mutating_input_sequence_cannot_change_configured_features() -> None:
    requested = ["dt", "dy"]
    vc = VectorChain(features=requested)

    requested.append("theta")

    assert vc.features == ("dt", "dy")


def test_delta_r_is_available_and_zero_for_first_vector() -> None:
    vc = VectorChain(
        tolerance=0.0,
        features=("dt", "dy", "r", "delta_theta", "delta_r"),
    )

    vectors = vc.fit_transform([0.0, 1.0, 2.0, 2.0, 2.0])

    first_radius = np.sqrt(8.0)
    np.testing.assert_allclose(
        vectors,
        [
            [2.0, 2.0, first_radius, 0.0, 0.0],
            [2.0, 0.0, 2.0, -np.pi / 4.0, 2.0 - first_radius],
        ],
    )


def test_feature_selection_never_changes_boundaries() -> None:
    signal = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 3.0]
    minimal = VectorChain(tolerance=0.1, features=("dt", "dy"))
    complete = VectorChain(tolerance=0.1, features=SUPPORTED_FEATURES)

    minimal.fit_transform(signal)
    complete.fit_transform(signal)

    assert minimal.segments_ == complete.segments_
    np.testing.assert_array_equal(minimal.segment_boundaries_, complete.segment_boundaries_)


def test_feature_helpers_return_canonical_validated_shapes() -> None:
    assert validate_feature_names(list(DEFAULT_FEATURES)) == DEFAULT_FEATURES
    matrix = compute_feature_matrix([], DEFAULT_FEATURES)

    assert matrix.dtype == np.float64
    assert matrix.shape == (0, len(DEFAULT_FEATURES))
