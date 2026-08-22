"""Metrics used to evaluate VectorChain reconstruction and compression."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean absolute reconstruction error."""

    observed, reconstructed = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(observed - reconstructed)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared reconstruction error."""

    observed, reconstructed = _validate_pair(y_true, y_pred)
    residual = observed - reconstructed
    return float(np.sqrt(np.mean(residual * residual)))


def compression_factor(n_points: int, n_vectors: int) -> float:
    """Return the structural sequence-length factor ``n_points / n_vectors``."""

    points = _validate_count(n_points, "n_points")
    vectors = _validate_count(n_vectors, "n_vectors")
    return points / vectors


def retention_fraction(n_points: int, n_vectors: int) -> float:
    """Return the retained sequence-length fraction ``n_vectors / n_points``."""

    points = _validate_count(n_points, "n_points")
    vectors = _validate_count(n_vectors, "n_vectors")
    return vectors / points


def _validate_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = _validate_metric_array(y_true, "y_true")
    reconstructed = _validate_metric_array(y_pred, "y_pred")
    if observed.shape != reconstructed.shape:
        msg = "y_true and y_pred must have the same shape"
        raise ValueError(msg)
    return observed, reconstructed


def _validate_metric_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be a one-dimensional real numeric array"
        raise ValueError(msg) from error
    if array.ndim != 1 or array.size == 0:
        msg = f"{name} must be a non-empty one-dimensional array"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        msg = f"{name} must contain real numeric values"
        raise TypeError(msg)
    validated = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return validated


def _validate_count(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        msg = f"{name} must be a positive integer"
        raise TypeError(msg)
    validated = int(value)
    if validated <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return validated
