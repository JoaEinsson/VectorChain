"""Simple sequence baselines for VectorChain comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorchain.features import (
    DEFAULT_FEATURES,
    FeatureName,
    compute_features_from_displacements,
    validate_feature_names,
)


@dataclass(frozen=True, slots=True)
class FixedSegmentation:
    """Vectors and articulated boundaries from predetermined segment durations."""

    vectors: NDArray[np.float64]
    boundaries: NDArray[np.int64]
    feature_names: tuple[FeatureName, ...]


def raw_values(x: ArrayLike) -> NDArray[np.float64]:
    """Return scalar observations as a two-dimensional sequence."""

    values = _validate_series(x, minimum_size=1)
    return values[:, np.newaxis].copy()


def normalized_raw_values(x: ArrayLike) -> NDArray[np.float64]:
    """Return per-series population z-scores as a column sequence.

    A constant input maps to zeros because its centered signal has no scale.
    """

    values = _validate_series(x, minimum_size=1)
    centered = values - np.mean(values)
    scale = float(np.std(values))
    if scale == 0.0:
        return np.zeros((values.size, 1), dtype=np.float64)
    return (centered / scale)[:, np.newaxis]


def first_difference(x: ArrayLike) -> NDArray[np.float64]:
    """Return first differences as a one-column sequence."""

    values = _validate_series(x, minimum_size=2)
    return np.diff(values)[:, np.newaxis]


def first_second_difference(x: ArrayLike) -> NDArray[np.float64]:
    """Return aligned first- and second-difference columns.

    The first difference is trimmed at the beginning so both columns refer to
    indices ``2, ..., n - 1`` of the original series.
    """

    values = _validate_series(x, minimum_size=3)
    first = np.diff(values)
    second = np.diff(first)
    return np.column_stack((first[1:], second))


def fixed_linear_segmentation(
    x: ArrayLike,
    *,
    segment_length: int,
    features: Sequence[str] = DEFAULT_FEATURES,
) -> FixedSegmentation:
    """Create articulated linear vectors at predetermined interval counts.

    ``segment_length`` counts sample intervals, not points. The terminal segment
    may be shorter so every observation remains covered exactly once apart from
    shared articulation endpoints.
    """

    values = _validate_series(x, minimum_size=2)
    duration = _validate_segment_length(segment_length)
    feature_names = validate_feature_names(features)
    starts = np.arange(0, values.size - 1, duration, dtype=np.int64)
    ends = np.minimum(starts + duration, values.size - 1)
    boundaries = np.column_stack((starts, ends)).astype(np.int64, copy=False)
    dt = (ends - starts).astype(np.float64)
    dy = values[ends] - values[starts]
    vectors = compute_features_from_displacements(dt, dy, feature_names)
    boundaries.flags.writeable = False
    vectors.flags.writeable = False
    return FixedSegmentation(vectors, boundaries, feature_names)


def _validate_series(x: ArrayLike, *, minimum_size: int) -> NDArray[np.float64]:
    try:
        values = np.asarray(x)
    except (TypeError, ValueError) as error:
        msg = "x must be a one-dimensional real numeric series"
        raise ValueError(msg) from error
    if values.ndim != 1 or values.size < minimum_size:
        msg = f"x must be one-dimensional with at least {minimum_size} observations"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        msg = "x must contain real numeric observations"
        raise TypeError(msg)
    validated = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "x must contain only finite observations"
        raise ValueError(msg)
    return validated


def _validate_segment_length(segment_length: int) -> int:
    if isinstance(segment_length, (bool, np.bool_)) or not isinstance(segment_length, Integral):
        msg = "segment_length must be a positive integer"
        raise TypeError(msg)
    validated = int(segment_length)
    if validated <= 0:
        msg = "segment_length must be a positive integer"
        raise ValueError(msg)
    return validated
