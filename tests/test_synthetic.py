"""Tests for deterministic synthetic signal generation."""

from collections.abc import Callable

import numpy as np
import pytest

from vectorchain import (
    generate_chirp,
    generate_first_order_response,
    generate_piecewise_linear,
    generate_ramp,
    generate_regime_change,
    generate_second_order_response,
    generate_sine,
)

GeneratorFunction = Callable[..., np.ndarray]

GENERATORS: tuple[GeneratorFunction, ...] = (
    generate_sine,
    generate_chirp,
    generate_ramp,
    generate_piecewise_linear,
    generate_first_order_response,
    generate_second_order_response,
    generate_regime_change,
)


@pytest.mark.parametrize("generator", GENERATORS)
def test_every_generator_is_reproducible_from_an_integer_seed(
    generator: GeneratorFunction,
) -> None:
    first = generator(rng=1729, n_points=64, amplitude=1.7, offset=-0.2, noise_std=0.03)
    second = generator(rng=1729, n_points=64, amplitude=1.7, offset=-0.2, noise_std=0.03)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (64,)
    assert first.dtype == np.float64
    assert np.all(np.isfinite(first))


def test_generator_object_advances_reproducibly_when_noise_is_enabled() -> None:
    rng = np.random.default_rng(42)
    first = generate_sine(rng=rng, n_points=32, noise_std=0.1)
    second = generate_sine(rng=rng, n_points=32, noise_std=0.1)

    replay = np.random.default_rng(42)
    replay_first = generate_sine(rng=replay, n_points=32, noise_std=0.1)
    replay_second = generate_sine(rng=replay, n_points=32, noise_std=0.1)

    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, replay_first)
    np.testing.assert_array_equal(second, replay_second)


def test_noise_free_generation_does_not_advance_supplied_rng() -> None:
    used = np.random.default_rng(7)
    control = np.random.default_rng(7)

    generate_ramp(rng=used, n_points=10, noise_std=0.0)

    assert used.random() == control.random()


def test_generation_never_uses_numpy_legacy_global_rng() -> None:
    previous_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected = np.random.random()
        np.random.seed(1234)

        generate_sine(rng=99, n_points=16, noise_std=0.2)

        assert np.random.random() == expected
    finally:
        np.random.set_state(previous_state)


def test_canonical_shapes_have_documented_endpoints_and_scale() -> None:
    sine = generate_sine(rng=1, n_points=5, amplitude=2.0, offset=3.0, frequency=1.0)
    ramp = generate_ramp(rng=1, n_points=5, amplitude=2.0, offset=3.0)
    first_order = generate_first_order_response(rng=1, n_points=20, offset=-1.0)
    second_order = generate_second_order_response(rng=1, n_points=100)
    regime = generate_regime_change(
        rng=1,
        n_points=11,
        amplitude=2.0,
        offset=1.0,
        frequency_before=2.0,
        frequency_after=2.0,
        change_fraction=0.5,
        level_shift=0.5,
    )

    np.testing.assert_allclose(sine, [3.0, 5.0, 3.0, 1.0, 3.0], atol=1e-14)
    np.testing.assert_array_equal(ramp, [3.0, 3.5, 4.0, 4.5, 5.0])
    assert first_order[0] == -1.0
    assert first_order[-1] > -0.01
    assert second_order[0] == 0.0
    assert np.max(second_order) > 1.0
    assert regime[5] == pytest.approx(2.0)


def test_piecewise_linear_generator_passes_through_fixed_knots() -> None:
    signal = generate_piecewise_linear(rng=5, n_points=21, amplitude=2.0, offset=1.0)

    np.testing.assert_allclose(signal[[0, 4, 9, 14, 20]], [1.0, 2.6, 0.2, 3.0, 1.4])


def test_chirp_phase_matches_integrated_linear_frequency() -> None:
    signal = generate_chirp(
        rng=8,
        n_points=5,
        start_frequency=1.0,
        end_frequency=3.0,
        phase=np.pi / 4.0,
    )
    time = np.linspace(0.0, 1.0, 5)
    expected = np.sin(2.0 * np.pi * (time + time**2) + np.pi / 4.0)

    np.testing.assert_allclose(signal, expected)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rng": True}, TypeError),
        ({"rng": "1729"}, TypeError),
        ({"rng": 1, "n_points": True}, TypeError),
        ({"rng": 1, "n_points": 1}, ValueError),
        ({"rng": 1, "amplitude": "1"}, TypeError),
        ({"rng": 1, "amplitude": np.inf}, ValueError),
        ({"rng": 1, "offset": np.nan}, ValueError),
        ({"rng": 1, "noise_std": True}, TypeError),
        ({"rng": 1, "noise_std": -0.1}, ValueError),
        ({"rng": 1, "frequency": -1.0}, ValueError),
        ({"rng": 1, "phase": np.inf}, ValueError),
    ],
)
def test_sine_rejects_invalid_common_parameters(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        generate_sine(**kwargs)


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (generate_chirp, {"start_frequency": -1.0}),
        (generate_chirp, {"end_frequency": -1.0}),
        (generate_first_order_response, {"time_constant": 0.0}),
        (generate_second_order_response, {"natural_frequency": 0.0}),
        (generate_second_order_response, {"damping_ratio": 0.0}),
        (generate_second_order_response, {"damping_ratio": 1.0}),
        (generate_second_order_response, {"damping_ratio": np.nan}),
        (generate_regime_change, {"frequency_before": -1.0}),
        (generate_regime_change, {"frequency_after": -1.0}),
        (generate_regime_change, {"change_fraction": 0.0}),
        (generate_regime_change, {"change_fraction": 1.0}),
        (generate_regime_change, {"level_shift": np.inf}),
    ],
)
def test_generator_specific_parameters_are_validated(
    generator: GeneratorFunction, kwargs: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        generator(rng=1, **kwargs)


def test_rng_is_an_explicit_required_argument() -> None:
    with pytest.raises(TypeError, match="rng"):
        generate_sine()
