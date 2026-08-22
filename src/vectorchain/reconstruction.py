"""Piecewise-linear reconstruction for fitted VectorChain representations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorchain.features import FeatureName


def reconstruct_vector_chain(
    vectors: ArrayLike,
    feature_names: Sequence[FeatureName],
    boundaries: NDArray[np.int64],
    *,
    initial_value: float,
    n_samples: int,
) -> NDArray[np.float64]:
    """Reconstruct samples from fitted boundaries and the ``dt``/``dy`` columns.

    The fitted boundaries provide the structural layout, while the supplied
    displacements determine the reconstructed endpoint values. Adjacent segments
    share one hinge sample, which is emitted only once.
    """

    matrix = _validate_vectors(vectors, len(feature_names))
    _validate_structure(matrix, feature_names, boundaries, n_samples)

    dt_index = feature_names.index("dt")
    dy_index = feature_names.index("dy")
    reconstructed = np.empty(n_samples, dtype=np.float64)
    reconstructed[0] = initial_value
    segment_start_value = initial_value

    for row_index, (start, end) in enumerate(boundaries):
        start_index = int(start)
        end_index = int(end)
        duration = end_index - start_index
        displacement = float(matrix[row_index, dy_index])
        segment_end_value = segment_start_value + displacement
        first_offset = 0 if row_index == 0 else 1
        offsets = np.arange(first_offset, duration + 1, dtype=np.float64)
        reconstructed[start_index + first_offset : end_index + 1] = (
            segment_start_value + offsets * displacement / float(matrix[row_index, dt_index])
        )
        reconstructed[end_index] = segment_end_value
        segment_start_value = segment_end_value

    return reconstructed


def _validate_vectors(vectors: ArrayLike, n_features: int) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(vectors)
    except (TypeError, ValueError) as error:
        msg = "Z must be a two-dimensional real numeric array"
        raise ValueError(msg) from error
    if matrix.ndim != 2:
        msg = "Z must be a two-dimensional real numeric array"
        raise ValueError(msg)
    if matrix.shape[1] != n_features:
        msg = f"Z must have exactly {n_features} columns"
        raise ValueError(msg)
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        msg = "Z must contain real numeric values"
        raise TypeError(msg)
    validated = matrix.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "Z must contain only finite values"
        raise ValueError(msg)
    return validated


def _validate_structure(
    vectors: NDArray[np.float64],
    feature_names: Sequence[FeatureName],
    boundaries: NDArray[np.int64],
    n_samples: int,
) -> None:
    if vectors.shape[0] != boundaries.shape[0]:
        msg = f"Z must have exactly {boundaries.shape[0]} rows for this fitted chain"
        raise ValueError(msg)

    durations = boundaries[:, 1] - boundaries[:, 0]
    encoded_durations = vectors[:, feature_names.index("dt")]
    if not np.array_equal(encoded_durations, durations):
        msg = "the dt column of Z must match the fitted segment boundaries"
        raise ValueError(msg)

    if boundaries.shape[0] == 0 or boundaries[0, 0] != 0 or boundaries[-1, 1] != n_samples - 1:
        msg = "fitted boundaries do not cover the reconstructed sample range"
        raise RuntimeError(msg)
    if boundaries.shape[0] > 1 and not np.array_equal(boundaries[1:, 0], boundaries[:-1, 1]):
        msg = "fitted boundaries must form a continuous articulated chain"
        raise RuntimeError(msg)
