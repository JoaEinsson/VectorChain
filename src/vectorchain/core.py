"""Causal online segmentation for scalar time series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorchain.features import (
    DEFAULT_FEATURES,
    FeatureName,
    compute_feature_matrix,
    validate_feature_names,
)

_ScalarT = TypeVar("_ScalarT", np.float64, np.int64)


@dataclass(frozen=True, slots=True)
class Segment:
    """An immutable finalized segment.

    Parameters
    ----------
    start, end:
        Inclusive sample indices. Consecutive segments share their hinge index.
    start_value, end_value:
        Observed scalar values at the segment endpoints.
    emitted_at:
        Index of the observation that caused online emission. ``None`` denotes
        explicit end-of-stream finalization.
    """

    start: int
    end: int
    start_value: float
    end_value: float
    emitted_at: int | None

    def __post_init__(self) -> None:
        """Reject structurally invalid public segment instances."""

        if isinstance(self.start, (bool, np.bool_)) or not isinstance(
            self.start, (int, np.integer)
        ):
            msg = "segment indices must be integers"
            raise TypeError(msg)
        if isinstance(self.end, (bool, np.bool_)) or not isinstance(self.end, (int, np.integer)):
            msg = "segment indices must be integers"
            raise TypeError(msg)
        if self.emitted_at is not None and (
            isinstance(self.emitted_at, (bool, np.bool_))
            or not isinstance(self.emitted_at, (int, np.integer))
        ):
            msg = "emitted_at must be an integer or None"
            raise TypeError(msg)
        if (
            isinstance(self.start_value, (bool, np.bool_))
            or not isinstance(self.start_value, Real)
            or isinstance(self.end_value, (bool, np.bool_))
            or not isinstance(self.end_value, Real)
        ):
            msg = "segment endpoint values must be finite real numbers"
            raise TypeError(msg)
        start_value = float(self.start_value)
        end_value = float(self.end_value)

        start = int(self.start)
        end = int(self.end)
        emitted_at = None if self.emitted_at is None else int(self.emitted_at)
        if start < 0 or end <= start:
            msg = "a segment must satisfy 0 <= start < end"
            raise ValueError(msg)
        if not np.isfinite(start_value) or not np.isfinite(end_value):
            msg = "segment endpoint values must be finite"
            raise ValueError(msg)
        if emitted_at is not None and emitted_at <= end:
            msg = "emitted_at must occur after the segment endpoint"
            raise ValueError(msg)

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "start_value", start_value)
        object.__setattr__(self, "end_value", end_value)
        object.__setattr__(self, "emitted_at", emitted_at)

    @property
    def dt(self) -> int:
        """Return the segment duration in sample intervals."""

        return self.end - self.start

    @property
    def dy(self) -> float:
        """Return the vertical displacement between endpoints."""

        return self.end_value - self.start_value


class VectorChain:
    """Causally segment a scalar time series into an adaptive vector chain.

    Parameters
    ----------
    tolerance:
        Maximum absolute vertical residual permitted for the endpoint chord.
    causal:
        Must be ``True`` in the MVP. Offline segmentation is intentionally absent.
    min_segment_length:
        Minimum number of points accepted before a normal threshold-based closure.
    features:
        Ordered output columns. ``dt`` and ``dy`` are required. Feature selection
        is applied only after segmentation and therefore cannot change boundaries.
    """

    def __init__(
        self,
        tolerance: float = 0.03,
        causal: bool = True,
        min_segment_length: int = 2,
        features: Sequence[str] = DEFAULT_FEATURES,
    ) -> None:
        if not isinstance(causal, bool):
            msg = "causal must be a bool"
            raise TypeError(msg)
        if not causal:
            msg = "only causal=True is implemented in the MVP"
            raise NotImplementedError(msg)

        self.tolerance = self._validate_tolerance(tolerance)
        self.causal = causal
        self.min_segment_length = self._validate_min_segment_length(min_segment_length)
        self._features = validate_feature_names(features)
        self.reset()

    @property
    def features(self) -> tuple[FeatureName, ...]:
        """Return the immutable configured feature selection."""

        return self._features

    @property
    def feature_names_(self) -> tuple[FeatureName, ...]:
        """Return feature names in the exact order used by ``vectors_``."""

        return self.features

    @property
    def segments_(self) -> tuple[Segment, ...]:
        """Return an immutable snapshot of finalized segments."""

        return tuple(self._segments)

    @property
    def is_finalized_(self) -> bool:
        """Whether end-of-stream finalization has occurred."""

        return self._is_finalized

    @property
    def open_segment_boundary_(self) -> tuple[int, int] | None:
        """Return the provisional open boundary, or ``None`` after finalization."""

        if self._open_start is None:
            return None
        return (self._open_start, self.n_samples_ - 1)

    def reset(self) -> VectorChain:
        """Clear all stream state and finalized outputs."""

        self._segments: list[Segment] = []
        self._open_values: list[float] = []
        self._open_start: int | None = None
        self._is_finalized = False

        self.n_samples_ = 0
        self.initial_value_: float | None = None
        self.vectors_ = self._readonly(np.empty((0, len(self.features)), dtype=np.float64))
        self.segment_boundaries_ = self._readonly(np.empty((0, 2), dtype=np.int64))
        return self

    def update(self, value: float) -> tuple[Segment, ...]:
        """Consume one observation and return any segment emitted at this step.

        Only the observations consumed by this instance can affect the transition.
        Calling ``update`` after ``finalize`` is an error; call ``reset`` first.
        """

        if self._is_finalized:
            msg = "cannot update a finalized stream; call reset() first"
            raise RuntimeError(msg)

        sample = self._validate_sample(value)
        index = self.n_samples_

        if index == 0:
            self.initial_value_ = sample
            self._open_start = 0
            self._open_values.append(sample)
            self.n_samples_ = 1
            return ()

        if len(self._open_values) == 1:
            self._open_values.append(sample)
            self.n_samples_ += 1
            return ()

        candidate = (*self._open_values, sample)
        must_grow = len(candidate) <= self.min_segment_length
        within_tolerance = self._chord_error(candidate) <= self.tolerance
        if must_grow or within_tolerance:
            self._open_values.append(sample)
            self.n_samples_ += 1
            return ()

        segment = self._make_open_segment(emitted_at=index)
        self._segments.append(segment)
        previous_endpoint = self._open_values[-1]
        self._open_start = index - 1
        self._open_values = [previous_endpoint, sample]
        self.n_samples_ += 1
        self._refresh_outputs()
        return (segment,)

    def finalize(self) -> tuple[Segment, ...]:
        """Emit the terminal open segment after an explicit end-of-stream signal."""

        if self._is_finalized:
            return ()
        if self.n_samples_ < 2:
            msg = "at least two observations are required to finalize a vector chain"
            raise ValueError(msg)

        segment = self._make_open_segment(emitted_at=None)
        self._segments.append(segment)
        self._open_values = []
        self._open_start = None
        self._is_finalized = True
        self._refresh_outputs()
        return (segment,)

    def fit_transform(self, x: ArrayLike) -> NDArray[np.float64]:
        """Reset, consume a one-dimensional series causally, and return its vectors."""

        self.reset()
        try:
            values = self._validate_series(x)
            for value in values:
                self.update(float(value))
            self.finalize()
        except Exception:
            self.reset()
            raise
        return self.vectors_.copy()

    @staticmethod
    def _validate_tolerance(tolerance: float) -> float:
        if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
            msg = "tolerance must be a finite non-negative real number"
            raise TypeError(msg)
        validated = float(tolerance)
        if not np.isfinite(validated) or validated < 0:
            msg = "tolerance must be finite and non-negative"
            raise ValueError(msg)
        return validated

    @staticmethod
    def _validate_min_segment_length(min_segment_length: int) -> int:
        if isinstance(min_segment_length, (bool, np.bool_)) or not isinstance(
            min_segment_length, (int, np.integer)
        ):
            msg = "min_segment_length must be an integer greater than or equal to 2"
            raise TypeError(msg)
        validated = int(min_segment_length)
        if validated < 2:
            msg = "min_segment_length must be greater than or equal to 2"
            raise ValueError(msg)
        return validated

    @staticmethod
    def _validate_sample(value: float) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            msg = "observations must be finite real numbers"
            raise TypeError(msg)
        sample = float(value)
        if not np.isfinite(sample):
            msg = "observations must be finite real numbers"
            raise ValueError(msg)
        return sample

    @staticmethod
    def _validate_series(x: ArrayLike) -> NDArray[np.float64]:
        try:
            values = np.asarray(x)
        except (TypeError, ValueError) as error:
            msg = "x must be a one-dimensional numeric series"
            raise ValueError(msg) from error
        if values.ndim != 1:
            msg = "x must be a one-dimensional numeric series"
            raise ValueError(msg)
        if values.size < 2:
            msg = "x must contain at least two observations"
            raise ValueError(msg)
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            msg = "x must contain real numeric observations"
            raise TypeError(msg)
        return values.astype(np.float64, copy=False)

    @staticmethod
    def _chord_error(values: Sequence[float]) -> float:
        observed = np.asarray(values, dtype=np.float64)
        step = (observed[-1] - observed[0]) / (observed.size - 1)
        reconstructed = observed[0] + np.arange(observed.size, dtype=np.float64) * step
        return float(np.max(np.abs(observed - reconstructed)))

    def _make_open_segment(self, emitted_at: int | None) -> Segment:
        if self._open_start is None or len(self._open_values) < 2:
            msg = "internal error: no valid open segment"
            raise RuntimeError(msg)
        return Segment(
            start=self._open_start,
            end=self.n_samples_ - 1,
            start_value=self._open_values[0],
            end_value=self._open_values[-1],
            emitted_at=emitted_at,
        )

    def _refresh_outputs(self) -> None:
        boundaries = np.asarray(
            [(segment.start, segment.end) for segment in self._segments], dtype=np.int64
        ).reshape((-1, 2))
        vectors = compute_feature_matrix(self._segments, self.features)
        self.segment_boundaries_ = self._readonly(boundaries)
        self.vectors_ = self._readonly(vectors)

    @staticmethod
    def _readonly(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
        array.flags.writeable = False
        return array
