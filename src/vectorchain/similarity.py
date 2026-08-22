"""Sequence standardization, DTW distance, and nearest-neighbor retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class FeatureStandardizer:
    """Column scales fitted exclusively from reference sequences."""

    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]

    @classmethod
    def fit(cls, sequences: Sequence[ArrayLike]) -> FeatureStandardizer:
        """Fit population mean and scale over all reference sequence steps."""

        validated = _validate_sequence_collection(sequences)
        combined = np.concatenate(validated, axis=0)
        mean = np.mean(combined, axis=0)
        scale = np.std(combined, axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        mean = np.asarray(mean, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        mean.flags.writeable = False
        scale.flags.writeable = False
        return cls(mean, scale)

    def transform(self, sequence: ArrayLike) -> NDArray[np.float64]:
        """Standardize a sequence without updating gallery statistics."""

        validated = _validate_sequence(sequence)
        if validated.shape[1] != self.mean_.size:
            msg = f"sequence must have exactly {self.mean_.size} features"
            raise ValueError(msg)
        return np.asarray((validated - self.mean_) / self.scale_, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Neighbor:
    """Index and deterministic distance of one ranked reference sequence."""

    index: int
    distance: float


def dtw_distance(left: ArrayLike, right: ArrayLike, *, window: int | None = None) -> float:
    """Return path-length-normalized DTW distance with local feature RMS."""

    first = _validate_sequence(left)
    second = _validate_sequence(right)
    if first.shape[1] != second.shape[1]:
        msg = "left and right must have the same number of features"
        raise ValueError(msg)
    width = _validate_window(window, first.shape[0], second.shape[0])
    n_right = second.shape[0]
    previous_cost = np.full(n_right + 1, np.inf, dtype=np.float64)
    previous_steps = np.zeros(n_right + 1, dtype=np.int64)
    previous_cost[0] = 0.0

    for left_index in range(1, first.shape[0] + 1):
        current_cost = np.full(n_right + 1, np.inf, dtype=np.float64)
        current_steps = np.zeros(n_right + 1, dtype=np.int64)
        start = max(1, left_index - width)
        stop = min(n_right, left_index + width)
        local_costs = np.sqrt(np.mean((second - first[left_index - 1]) ** 2, axis=1))
        for right_index in range(start, stop + 1):
            best_cost = previous_cost[right_index - 1]
            best_steps = previous_steps[right_index - 1]
            if previous_cost[right_index] < best_cost:
                best_cost = previous_cost[right_index]
                best_steps = previous_steps[right_index]
            if current_cost[right_index - 1] < best_cost:
                best_cost = current_cost[right_index - 1]
                best_steps = current_steps[right_index - 1]
            current_cost[right_index] = best_cost + local_costs[right_index - 1]
            current_steps[right_index] = best_steps + 1
        previous_cost = current_cost
        previous_steps = current_steps

    if not np.isfinite(previous_cost[n_right]) or previous_steps[n_right] == 0:
        msg = "no valid DTW path exists for the configured window"
        raise RuntimeError(msg)
    return float(previous_cost[n_right] / previous_steps[n_right])


def nearest_neighbors(
    query: ArrayLike,
    references: Sequence[ArrayLike],
    *,
    k: int = 1,
    window: int | None = None,
) -> tuple[Neighbor, ...]:
    """Rank reference sequences by normalized DTW distance."""

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        msg = "k must be a positive integer"
        raise TypeError(msg)
    validated_k = int(k)
    if validated_k <= 0:
        msg = "k must be a positive integer"
        raise ValueError(msg)
    if len(references) == 0:
        msg = "references must contain at least one sequence"
        raise ValueError(msg)
    if validated_k > len(references):
        msg = "k cannot exceed the number of references"
        raise ValueError(msg)

    distances = np.fromiter(
        (dtw_distance(query, reference, window=window) for reference in references),
        dtype=np.float64,
        count=len(references),
    )
    ranking = np.argsort(distances, kind="stable")[:validated_k]
    return tuple(Neighbor(int(index), float(distances[index])) for index in ranking)


def _validate_sequence_collection(
    sequences: Sequence[ArrayLike],
) -> tuple[NDArray[np.float64], ...]:
    if len(sequences) == 0:
        msg = "sequences must contain at least one sequence"
        raise ValueError(msg)
    validated = tuple(_validate_sequence(sequence) for sequence in sequences)
    n_features = validated[0].shape[1]
    if any(sequence.shape[1] != n_features for sequence in validated[1:]):
        msg = "all sequences must have the same number of features"
        raise ValueError(msg)
    return validated


def _validate_sequence(sequence: ArrayLike) -> NDArray[np.float64]:
    try:
        values = np.asarray(sequence)
    except (TypeError, ValueError) as error:
        msg = "a sequence must be a non-empty one- or two-dimensional numeric array"
        raise ValueError(msg) from error
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        msg = "a sequence must be a non-empty one- or two-dimensional numeric array"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        msg = "sequences must contain real numeric values"
        raise TypeError(msg)
    validated = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "sequences must contain only finite values"
        raise ValueError(msg)
    return validated


def _validate_window(window: int | None, n_left: int, n_right: int) -> int:
    if window is None:
        return max(n_left, n_right)
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
        msg = "window must be a non-negative integer or None"
        raise TypeError(msg)
    validated = int(window)
    if validated < 0:
        msg = "window must be a non-negative integer or None"
        raise ValueError(msg)
    return max(validated, abs(n_left - n_right))
