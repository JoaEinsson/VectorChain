"""Deterministic synthetic scalar signals for VectorChain experiments."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

RngLike: TypeAlias = int | np.integer[Any] | np.random.Generator


def generate_sine(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    frequency: float = 3.0,
    phase: float = 0.0,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a sine wave over a normalized unit-time interval."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    cycles = _validate_non_negative_real(frequency, "frequency")
    phase_radians = _validate_finite_real(phase, "phase")
    clean = np.sin(2.0 * np.pi * cycles * time + phase_radians)
    return _finish(clean, scale, baseline, noise, generator)


def generate_chirp(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    start_frequency: float = 1.0,
    end_frequency: float = 8.0,
    phase: float = 0.0,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a linear-frequency chirp over normalized unit time."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    start = _validate_non_negative_real(start_frequency, "start_frequency")
    end = _validate_non_negative_real(end_frequency, "end_frequency")
    phase_radians = _validate_finite_real(phase, "phase")
    chirp_phase = 2.0 * np.pi * (start * time + 0.5 * (end - start) * time**2)
    clean = np.sin(chirp_phase + phase_radians)
    return _finish(clean, scale, baseline, noise, generator)


def generate_ramp(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a ramp whose total rise equals ``amplitude``."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    return _finish(time, scale, baseline, noise, generator)


def generate_piecewise_linear(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a fixed five-knot piecewise-linear reference shape."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    knot_times = np.array([0.0, 0.2, 0.45, 0.7, 1.0], dtype=np.float64)
    knot_values = np.array([0.0, 0.8, -0.4, 1.0, 0.2], dtype=np.float64)
    clean = np.interp(time, knot_times, knot_values)
    return _finish(clean, scale, baseline, noise, generator)


def generate_first_order_response(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    time_constant: float = 0.15,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate the unit-step response of a normalized first-order system."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    tau = _validate_positive_real(time_constant, "time_constant")
    clean = 1.0 - np.exp(-time / tau)
    return _finish(clean, scale, baseline, noise, generator)


def generate_second_order_response(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    natural_frequency: float = 3.0,
    damping_ratio: float = 0.15,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate an underdamped second-order unit-step response."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    frequency = _validate_positive_real(natural_frequency, "natural_frequency")
    damping = _validate_finite_real(damping_ratio, "damping_ratio")
    if not 0.0 < damping < 1.0:
        msg = "damping_ratio must satisfy 0 < damping_ratio < 1"
        raise ValueError(msg)

    omega_n = 2.0 * np.pi * frequency
    damping_term = np.sqrt(1.0 - damping**2)
    omega_d = omega_n * damping_term
    clean = 1.0 - np.exp(-damping * omega_n * time) * (
        np.cos(omega_d * time) + damping * np.sin(omega_d * time) / damping_term
    )
    return _finish(clean, scale, baseline, noise, generator)


def generate_regime_change(
    *,
    rng: RngLike,
    n_points: int = 1000,
    amplitude: float = 1.0,
    offset: float = 0.0,
    frequency_before: float = 2.0,
    frequency_after: float = 8.0,
    change_fraction: float = 0.5,
    level_shift: float = 0.5,
    noise_std: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a phase-continuous sine with frequency and level regime changes."""

    time, scale, baseline, noise, generator = _prepare(n_points, amplitude, offset, noise_std, rng)
    before = _validate_non_negative_real(frequency_before, "frequency_before")
    after = _validate_non_negative_real(frequency_after, "frequency_after")
    change = _validate_finite_real(change_fraction, "change_fraction")
    if not 0.0 < change < 1.0:
        msg = "change_fraction must satisfy 0 < change_fraction < 1"
        raise ValueError(msg)
    shift = _validate_finite_real(level_shift, "level_shift")

    accumulated_cycles = before * np.minimum(time, change) + after * np.maximum(time - change, 0.0)
    clean = np.sin(2.0 * np.pi * accumulated_cycles)
    clean = clean + np.where(time >= change, shift, 0.0)
    return _finish(clean, scale, baseline, noise, generator)


def _prepare(
    n_points: int,
    amplitude: float,
    offset: float,
    noise_std: float,
    rng: RngLike,
) -> tuple[
    NDArray[np.float64],
    float,
    float,
    float,
    np.random.Generator,
]:
    count = _validate_n_points(n_points)
    scale = _validate_finite_real(amplitude, "amplitude")
    baseline = _validate_finite_real(offset, "offset")
    noise = _validate_non_negative_real(noise_std, "noise_std")
    generator = _validate_rng(rng)
    return np.linspace(0.0, 1.0, count, dtype=np.float64), scale, baseline, noise, generator


def _finish(
    clean: NDArray[np.float64],
    amplitude: float,
    offset: float,
    noise_std: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    signal = amplitude * clean + offset
    if noise_std > 0.0:
        signal = signal + rng.normal(0.0, noise_std, size=clean.size)
    return np.asarray(signal, dtype=np.float64)


def _validate_n_points(n_points: int) -> int:
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral):
        msg = "n_points must be an integer greater than or equal to 2"
        raise TypeError(msg)
    validated = int(n_points)
    if validated < 2:
        msg = "n_points must be greater than or equal to 2"
        raise ValueError(msg)
    return validated


def _validate_finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a finite real number"
        raise TypeError(msg)
    validated = float(value)
    if not np.isfinite(validated):
        msg = f"{name} must be a finite real number"
        raise ValueError(msg)
    return validated


def _validate_non_negative_real(value: float, name: str) -> float:
    validated = _validate_finite_real(value, name)
    if validated < 0.0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    return validated


def _validate_positive_real(value: float, name: str) -> float:
    validated = _validate_finite_real(value, name)
    if validated <= 0.0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return validated


def _validate_rng(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer)):
        msg = "rng must be an integer seed or numpy.random.Generator"
        raise TypeError(msg)
    return np.random.default_rng(int(rng))
