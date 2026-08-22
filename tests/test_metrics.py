"""Tests for reconstruction and structural compression metrics."""

import numpy as np
import pytest

from vectorchain import compression_factor, mae, retention_fraction, rmse


def test_error_metrics_match_known_values() -> None:
    observed = [0.0, 1.0, 2.0]
    reconstructed = [0.0, 0.0, 1.0]

    assert mae(observed, reconstructed) == pytest.approx(2.0 / 3.0)
    assert rmse(observed, reconstructed) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_structural_compression_metrics_are_reciprocal() -> None:
    assert compression_factor(12, 3) == 4.0
    assert retention_fraction(12, 3) == 0.25


@pytest.mark.parametrize(
    ("observed", "reconstructed", "error"),
    [
        ([], [], ValueError),
        ([[0.0, 1.0]], [[0.0, 1.0]], ValueError),
        ([0.0], [0.0, 1.0], ValueError),
        ([0.0, np.nan], [0.0, 1.0], ValueError),
        ([0.0, 1.0], [0.0, np.inf], ValueError),
        (["zero"], ["zero"], TypeError),
        ([0.0 + 0.0j], [0.0 + 0.0j], TypeError),
    ],
)
def test_error_metrics_reject_invalid_inputs(
    observed: object, reconstructed: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        mae(observed, reconstructed)
    with pytest.raises(error):
        rmse(observed, reconstructed)


@pytest.mark.parametrize(
    ("n_points", "n_vectors", "error"),
    [
        (True, 1, TypeError),
        (1, False, TypeError),
        (1.5, 1, TypeError),
        (1, 1.5, TypeError),
        (0, 1, ValueError),
        (1, 0, ValueError),
        (-1, 1, ValueError),
        (1, -1, ValueError),
    ],
)
def test_compression_metrics_reject_invalid_counts(
    n_points: object, n_vectors: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        compression_factor(n_points, n_vectors)
    with pytest.raises(error):
        retention_fraction(n_points, n_vectors)
