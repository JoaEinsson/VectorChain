"""Unit tests for the causal VectorChain state machine."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vectorchain import DEFAULT_FEATURES, Segment, VectorChain


def test_initial_state_is_empty_and_open() -> None:
    vc = VectorChain()

    assert vc.feature_names_ == DEFAULT_FEATURES
    assert vc.segments_ == ()
    assert vc.open_segment_boundary_ is None
    assert not vc.is_finalized_
    assert vc.n_samples_ == 0
    assert vc.initial_value_ is None
    assert vc.compression_factor_ is None
    assert vc.compression_ratio_ is None
    assert vc.retention_fraction_ is None
    assert vc.reconstruction_error_ is None
    assert vc.vectors_.shape == (0, len(DEFAULT_FEATURES))
    assert vc.segment_boundaries_.shape == (0, 2)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tolerance": True}, TypeError),
        ({"tolerance": "0.1"}, TypeError),
        ({"tolerance": -0.1}, ValueError),
        ({"tolerance": np.nan}, ValueError),
        ({"causal": "yes"}, TypeError),
        ({"causal": False}, NotImplementedError),
        ({"min_segment_length": True}, TypeError),
        ({"min_segment_length": 2.5}, TypeError),
        ({"min_segment_length": 1}, ValueError),
    ],
)
def test_constructor_rejects_invalid_parameters(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        VectorChain(**kwargs)


def test_update_tracks_provisional_state_and_finalize_is_idempotent() -> None:
    vc = VectorChain(tolerance=0.0)

    assert vc.update(1.0) == ()
    assert vc.open_segment_boundary_ == (0, 0)
    assert vc.update(2.0) == ()
    assert vc.open_segment_boundary_ == (0, 1)

    emitted = vc.finalize()

    assert emitted == (Segment(0, 1, 1.0, 2.0, None),)
    assert vc.finalize() == ()
    assert vc.open_segment_boundary_ is None
    assert vc.is_finalized_
    with pytest.raises(RuntimeError, match="finalized stream"):
        vc.update(3.0)


@pytest.mark.parametrize("signal", [np.zeros(8), np.arange(8.0), 3.0 * np.arange(8.0) - 2.0])
def test_linear_signals_form_one_exact_segment(signal: np.ndarray) -> None:
    vc = VectorChain(tolerance=0.0)

    vectors = vc.fit_transform(signal)

    assert vc.segments_ == (Segment(0, 7, float(signal[0]), float(signal[-1]), None),)
    np.testing.assert_array_equal(vc.segment_boundaries_, [[0, 7]])
    assert vectors.shape == (1, 5)
    assert vectors[0, 0] == 7.0
    assert vectors[0, 1] == signal[-1] - signal[0]
    assert vc.initial_value_ == signal[0]
    assert vc.n_samples_ == signal.size


def test_piecewise_linear_signal_emits_articulated_segments() -> None:
    vc = VectorChain(tolerance=0.0)

    vectors = vc.fit_transform([0.0, 1.0, 2.0, 2.0, 2.0])

    expected_radius = np.sqrt(8.0)
    np.testing.assert_array_equal(vc.segment_boundaries_, [[0, 2], [2, 4]])
    np.testing.assert_allclose(
        vectors,
        [
            [2.0, 2.0, np.pi / 4.0, expected_radius, 0.0],
            [2.0, 0.0, 0.0, 2.0, -np.pi / 4.0],
        ],
    )
    assert vc.segments_[0].emitted_at == 3
    assert vc.segments_[1].emitted_at is None


def test_min_segment_length_can_force_acceptance_above_tolerance() -> None:
    short = VectorChain(tolerance=0.0, min_segment_length=2)
    forced = VectorChain(tolerance=0.0, min_segment_length=3)

    short.fit_transform([0.0, 1.0, 0.0])
    forced.fit_transform([0.0, 1.0, 0.0])

    np.testing.assert_array_equal(short.segment_boundaries_, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(forced.segment_boundaries_, [[0, 2]])


def test_reset_removes_all_information_from_previous_stream() -> None:
    vc = VectorChain(tolerance=0.0)
    vc.fit_transform([0.0, 1.0, 0.0])

    returned = vc.reset()

    assert returned is vc
    assert vc.segments_ == ()
    assert vc.n_samples_ == 0
    assert vc.initial_value_ is None
    assert vc.compression_factor_ is None
    assert vc.compression_ratio_ is None
    assert vc.retention_fraction_ is None
    assert vc.reconstruction_error_ is None
    assert vc.open_segment_boundary_ is None
    assert not vc.is_finalized_


def test_outputs_are_read_only_but_returned_transform_is_independent() -> None:
    vc = VectorChain(tolerance=0.0)

    transformed = vc.fit_transform([0.0, 1.0, 2.0])

    assert not vc.vectors_.flags.writeable
    assert not vc.segment_boundaries_.flags.writeable
    assert transformed.flags.writeable
    transformed[0, 0] = -1.0
    assert vc.vectors_[0, 0] == 2.0


@pytest.mark.parametrize(
    ("series", "error"),
    [
        ([1.0], ValueError),
        ([[1.0, 2.0]], ValueError),
        (["a", "b"], TypeError),
        ([1.0 + 1.0j, 2.0 + 0.0j], TypeError),
        ([0.0, np.nan], ValueError),
        ([0.0, np.inf], ValueError),
    ],
)
def test_fit_transform_rejects_invalid_series(series: object, error: type[Exception]) -> None:
    vc = VectorChain()

    with pytest.raises(error):
        vc.fit_transform(series)

    assert vc.segments_ == ()
    assert vc.n_samples_ == 0


def test_failed_fit_transform_clears_previous_fitted_state() -> None:
    vc = VectorChain()
    vc.fit_transform([0.0, 1.0])

    with pytest.raises(TypeError):
        vc.fit_transform(["a", "b"])

    assert vc.segments_ == ()
    assert vc.n_samples_ == 0
    assert vc.initial_value_ is None


def test_stream_rejects_invalid_observations_and_too_short_finalization() -> None:
    vc = VectorChain()

    with pytest.raises(TypeError):
        vc.update(True)
    with pytest.raises(TypeError):
        vc.update("1.0")
    with pytest.raises(ValueError, match="finite real"):
        vc.update(np.nan)
    with pytest.raises(ValueError, match="at least two"):
        vc.finalize()

    vc.update(1.0)
    with pytest.raises(ValueError, match="at least two"):
        vc.finalize()


def test_segment_is_normalized_validated_and_immutable() -> None:
    segment = Segment(np.int64(1), np.int64(3), np.float64(2.0), np.float64(5.0), np.int64(4))

    assert segment.dt == 2
    assert segment.dy == 3.0
    assert isinstance(segment.start, int)
    with pytest.raises(FrozenInstanceError):
        segment.end = 4


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((0.5, 2, 0.0, 1.0, None), TypeError),
        ((0, 2.5, 0.0, 1.0, None), TypeError),
        ((0, 2, 0.0, 1.0, 2.5), TypeError),
        ((0, 0, 0.0, 1.0, None), ValueError),
        ((0, 2, True, 1.0, None), TypeError),
        ((0, 2, "0.0", 1.0, None), TypeError),
        ((0, 2, np.nan, 1.0, None), ValueError),
        ((0, 2, 0.0, 1.0, 2), ValueError),
    ],
)
def test_segment_rejects_invalid_structure(
    args: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Segment(*args)
