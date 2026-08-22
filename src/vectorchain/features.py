"""Feature definitions and projection for finalized VectorChain segments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from vectorchain.core import Segment

FeatureName = Literal["dt", "dy", "theta", "r", "delta_theta", "delta_r"]

SUPPORTED_FEATURES: Final[tuple[FeatureName, ...]] = (
    "dt",
    "dy",
    "theta",
    "r",
    "delta_theta",
    "delta_r",
)

DEFAULT_FEATURES: Final[tuple[FeatureName, ...]] = (
    "dt",
    "dy",
    "theta",
    "r",
    "delta_theta",
)

_REQUIRED_FEATURES: Final[tuple[FeatureName, ...]] = ("dt", "dy")


def validate_feature_names(features: Sequence[str]) -> tuple[FeatureName, ...]:
    """Validate and normalize an ordered feature selection.

    The order is preserved because it defines the columns returned by
    :attr:`vectorchain.VectorChain.vectors_`.
    """

    if isinstance(features, str):
        msg = "features must be a sequence of names, not a single string"
        raise TypeError(msg)

    names = tuple(features)
    if not names:
        msg = "features must contain at least 'dt' and 'dy'"
        raise ValueError(msg)
    if any(not isinstance(name, str) for name in names):
        msg = "every feature name must be a string"
        raise TypeError(msg)

    duplicates = tuple(dict.fromkeys(name for name in names if names.count(name) > 1))
    if duplicates:
        msg = f"duplicate feature names are not allowed: {duplicates}"
        raise ValueError(msg)

    unknown = tuple(name for name in names if name not in SUPPORTED_FEATURES)
    if unknown:
        msg = f"unsupported feature names: {unknown}; supported names: {SUPPORTED_FEATURES}"
        raise ValueError(msg)

    missing = tuple(name for name in _REQUIRED_FEATURES if name not in names)
    if missing:
        msg = f"required feature names are missing: {missing}"
        raise ValueError(msg)

    return cast(tuple[FeatureName, ...], names)


def compute_feature_matrix(
    segments: Sequence[Segment], feature_names: tuple[FeatureName, ...]
) -> NDArray[np.float64]:
    """Compute selected features without affecting segment boundaries."""

    n_segments = len(segments)
    if n_segments == 0:
        return np.empty((0, len(feature_names)), dtype=np.float64)

    dt = np.fromiter(
        (segment.end - segment.start for segment in segments),
        dtype=np.float64,
        count=n_segments,
    )
    dy = np.fromiter(
        (segment.end_value - segment.start_value for segment in segments),
        dtype=np.float64,
        count=n_segments,
    )
    theta = np.arctan2(dy, dt)
    radius = np.hypot(dt, dy)

    delta_theta = np.zeros_like(theta)
    delta_radius = np.zeros_like(radius)
    if n_segments > 1:
        delta_theta[1:] = theta[1:] - theta[:-1]
        delta_radius[1:] = radius[1:] - radius[:-1]

    columns: dict[FeatureName, NDArray[np.float64]] = {
        "dt": dt,
        "dy": dy,
        "theta": theta,
        "r": radius,
        "delta_theta": delta_theta,
        "delta_r": delta_radius,
    }
    return np.column_stack(tuple(columns[name] for name in feature_names))
