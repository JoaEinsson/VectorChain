"""Pre-registered K7 signals and causally matched representation designs."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from vectorchain import (
    CommittedLink,
    RevisableVectorChain,
    TailEvent,
    WorkingVersion,
)

DEVELOPMENT_SEEDS = (11, 22)
MECHANISM_NAMES = (
    "frequency_modulation",
    "baseline_modulation",
    "crest_asymmetry_modulation",
)
REPRESENTATION_NAMES = (
    "immutable_absolute",
    "revisable_absolute",
    "revisable_spatial",
    "revisable_temporal",
    "raw_matched",
    "persistence",
)
TRAINED_REPRESENTATION_NAMES = REPRESENTATION_NAMES[:-1]
HORIZONS = (1, 8, 32)
N_POINTS = 4096
NOISE_STD = 0.02
ORIGIN_STRIDE = 4
RAW_MATCHED_STEPS = 16
VECTOR_STEPS = 4
N_INPUT_SCALARS = 17
N_OUTPUTS = 3
N_PREDICTIVE_PARAMETERS = N_OUTPUTS * (N_INPUT_SCALARS + 1)
TOLERANCE = 0.03
MIN_SEGMENT_LENGTH = 2


@dataclass(frozen=True, slots=True)
class K7Signal:
    """One isolated K7 oscillator and its analytic latent coordinate."""

    mechanism: str
    seed: int
    values: NDArray[np.float64]
    latent_name: str
    latent_coordinate: NDArray[np.float64]
    latent_derivative: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RepresentationDesign:
    """One causally aligned K7 input matrix and its payload accounting."""

    name: str
    inputs: NDArray[np.float64]
    feature_names: tuple[str, ...]
    input_steps: int
    scalar_elements: int
    predictive_parameters: int


@dataclass(frozen=True, slots=True)
class K7DesignBundle:
    """Shared origins, targets, audits and all six K7 representations."""

    signal: K7Signal
    lambda_revision: float
    lambda_bend: float
    origins: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    targets: NDArray[np.float64]
    link_ids: NDArray[np.int64]
    link_created_at: NDArray[np.int64]
    boundaries: NDArray[np.int64]
    condition_numbers: NDArray[np.float64]
    representations: tuple[RepresentationDesign, ...]
    versions: tuple[WorkingVersion, ...]
    committed: tuple[CommittedLink, ...]
    events: tuple[TailEvent, ...]

    def representation(self, name: str) -> RepresentationDesign:
        """Return one named representation or raise ``KeyError``."""

        for design in self.representations:
            if design.name == name:
                return design
        raise KeyError(name)


def generate_k7_signal(
    mechanism: str,
    *,
    seed: int,
    n_points: int = N_POINTS,
    noise_std: float = NOISE_STD,
) -> K7Signal:
    """Generate one pre-registered isolated oscillator condition.

    ``frequency_modulation`` uses ``phi[0] = 0`` and then the documented
    right-endpoint recurrence for samples ``1..n-1``. The returned derivative is
    analytic with respect to normalized time ``u`` and is not estimated from the
    noisy observations.
    """

    if mechanism not in MECHANISM_NAMES:
        msg = f"mechanism must be one of {MECHANISM_NAMES}"
        raise ValueError(msg)
    validated_seed = _validate_seed(seed)
    count = _validate_n_points(n_points)
    noise = _validate_noise_std(noise_std)
    normalized_time = np.linspace(0.0, 1.0, count, dtype=np.float64)
    modulation_phase = 2.0 * np.pi * 3.0 * normalized_time

    if mechanism == "frequency_modulation":
        latent_name = "f"
        latent = 20.0 - 12.0 * np.cos(modulation_phase)
        derivative = 72.0 * np.pi * np.sin(modulation_phase)
        phase = np.zeros(count, dtype=np.float64)
        phase[1:] = np.cumsum(2.0 * np.pi * latent[1:] / (count - 1))
        baseline = np.zeros(count, dtype=np.float64)
        asymmetry = np.zeros(count, dtype=np.float64)
    elif mechanism == "baseline_modulation":
        latent_name = "mu"
        latent = 1.0 - np.cos(modulation_phase)
        derivative = 6.0 * np.pi * np.sin(modulation_phase)
        phase = 2.0 * np.pi * 16.0 * normalized_time
        baseline = latent
        asymmetry = np.zeros(count, dtype=np.float64)
    else:
        latent_name = "kappa"
        latent = 0.225 * (1.0 - np.cos(modulation_phase))
        derivative = 1.35 * np.pi * np.sin(modulation_phase)
        phase = 2.0 * np.pi * 16.0 * normalized_time
        baseline = np.zeros(count, dtype=np.float64)
        asymmetry = latent

    clean = baseline + _oscillator(phase, asymmetry)
    if noise > 0.0:
        clean = clean + np.random.default_rng(validated_seed).normal(0.0, noise, size=count)
    return K7Signal(
        mechanism=mechanism,
        seed=validated_seed,
        values=_readonly_float(clean),
        latent_name=latent_name,
        latent_coordinate=_readonly_float(latent),
        latent_derivative=_readonly_float(derivative),
    )


def build_k7_designs(
    signal: K7Signal,
    *,
    lambda_revision: float,
    lambda_bend: float,
) -> K7DesignBundle:
    """Build all six K7 representations on one shared set of causal origins.

    An eligible origin is divisible by four, has exactly four working links,
    provides all 16 past raw increments, and leaves all three target horizons in
    bounds. Representation inputs are captured during online processing; targets
    are attached only after the causal state history has been built.
    """

    values = _validate_signal(signal)
    chain = RevisableVectorChain(
        tolerance=TOLERANCE,
        min_segment_length=MIN_SEGMENT_LENGTH,
        lambda_revision=lambda_revision,
        lambda_bend=lambda_bend,
    )
    origins: list[int] = []
    link_ids: list[tuple[int, ...]] = []
    link_created_at: list[tuple[int, ...]] = []
    boundaries: list[tuple[tuple[int, int], ...]] = []
    condition_numbers: list[float] = []
    inputs: dict[str, list[NDArray[np.float64]]] = {
        name: [] for name in TRAINED_REPRESENTATION_NAMES
    }

    maximum_horizon = max(HORIZONS)
    for origin, value in enumerate(values):
        version = chain.update(float(value))
        if not _eligible_origin(origin, version, values.size, maximum_horizon):
            continue

        origins.append(origin)
        link_ids.append(tuple(link.link_id for link in version.links))
        link_created_at.append(tuple(link.created_at for link in version.links))
        boundaries.append(tuple((link.start, link.end) for link in version.links))
        condition_numbers.append(version.condition_number)
        inputs["immutable_absolute"].append(_immutable_absolute(values, version))
        inputs["revisable_absolute"].append(_revisable_absolute(version))
        inputs["revisable_spatial"].append(_revisable_spatial(version))
        inputs["revisable_temporal"].append(_revisable_temporal(version))
        inputs["raw_matched"].append(_raw_matched(values, origin))

    if not origins:
        msg = "signal produced no origins eligible for all six K7 representations"
        raise ValueError(msg)

    origin_array = np.asarray(origins, dtype=np.int64)
    target_indices = origin_array[:, np.newaxis] + np.asarray(HORIZONS, dtype=np.int64)
    targets = values[target_indices] - values[origin_array, np.newaxis]
    feature_names = _feature_names()
    designs = (
        *(
            RepresentationDesign(
                name=name,
                inputs=_readonly_float(np.vstack(inputs[name])),
                feature_names=feature_names[name],
                input_steps=VECTOR_STEPS if name != "raw_matched" else RAW_MATCHED_STEPS,
                scalar_elements=N_INPUT_SCALARS,
                predictive_parameters=N_PREDICTIVE_PARAMETERS,
            )
            for name in TRAINED_REPRESENTATION_NAMES
        ),
        RepresentationDesign(
            name="persistence",
            inputs=_readonly_float(np.empty((origin_array.size, 0), dtype=np.float64)),
            feature_names=(),
            input_steps=0,
            scalar_elements=0,
            predictive_parameters=0,
        ),
    )
    return K7DesignBundle(
        signal=signal,
        lambda_revision=chain.lambda_revision,
        lambda_bend=chain.lambda_bend,
        origins=_readonly_int(origin_array),
        target_indices=_readonly_int(target_indices),
        targets=_readonly_float(targets),
        link_ids=_readonly_int(np.asarray(link_ids, dtype=np.int64)),
        link_created_at=_readonly_int(np.asarray(link_created_at, dtype=np.int64)),
        boundaries=_readonly_int(np.asarray(boundaries, dtype=np.int64)),
        condition_numbers=_readonly_float(np.asarray(condition_numbers, dtype=np.float64)),
        representations=designs,
        versions=chain.versions_,
        committed=chain.committed_,
        events=chain.events_,
    )


def _oscillator(phase: NDArray[np.float64], asymmetry: NDArray[np.float64]) -> NDArray[np.float64]:
    return (np.sin(phase) + asymmetry * np.sin(2.0 * phase + np.pi / 4.0)) / (
        1.0 + np.abs(asymmetry)
    )


def _eligible_origin(
    origin: int,
    version: WorkingVersion,
    n_points: int,
    maximum_horizon: int,
) -> bool:
    return (
        origin % ORIGIN_STRIDE == 0
        and origin >= RAW_MATCHED_STEPS
        and len(version.links) == VECTOR_STEPS
        and origin + maximum_horizon < n_points
    )


def _immutable_absolute(
    values: NDArray[np.float64], version: WorkingVersion
) -> NDArray[np.float64]:
    rows = []
    for link in version.links:
        dt = link.dt
        dy = float(values[link.end] - values[link.start])
        rows.append((dt, dy, atan2(dy, dt), hypot(dt, dy)))
    return _flatten_with_anchor(rows, values[version.observed_at])


def _revisable_absolute(version: WorkingVersion) -> NDArray[np.float64]:
    rows = [(link.dt, link.dy, link.theta, link.r) for link in version.links]
    return _flatten_with_anchor(rows, version.joints[-1].value)


def _revisable_spatial(version: WorkingVersion) -> NDArray[np.float64]:
    rows: list[tuple[float, float, float, float]] = []
    previous_theta: float | None = None
    previous_r: float | None = None
    for link in version.links:
        delta_theta = 0.0 if previous_theta is None else link.theta - previous_theta
        delta_r = 0.0 if previous_r is None else link.r - previous_r
        rows.append((link.dt, link.dy, delta_theta, delta_r))
        previous_theta = link.theta
        previous_r = link.r
    return _flatten_with_anchor(rows, version.joints[-1].value)


def _revisable_temporal(version: WorkingVersion) -> NDArray[np.float64]:
    rows = [(link.dt, link.dy, link.update_theta, link.update_r) for link in version.links]
    return _flatten_with_anchor(rows, version.joints[-1].value)


def _raw_matched(values: NDArray[np.float64], origin: int) -> NDArray[np.float64]:
    increments = np.diff(values[origin - RAW_MATCHED_STEPS : origin + 1])
    return np.concatenate((increments, np.asarray([values[origin]], dtype=np.float64)))


def _flatten_with_anchor(
    rows: list[tuple[float, float, float, float]], anchor: float
) -> NDArray[np.float64]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (VECTOR_STEPS, 4):
        msg = "a K7 vector representation requires exactly four links"
        raise ValueError(msg)
    return np.concatenate((matrix.ravel(), np.asarray([anchor], dtype=np.float64)))


def _feature_names() -> dict[str, tuple[str, ...]]:
    per_link = {
        "immutable_absolute": ("dt", "dy", "theta", "r"),
        "revisable_absolute": ("dt", "dy", "theta", "r"),
        "revisable_spatial": ("dt", "dy", "delta_theta_space", "delta_r_space"),
        "revisable_temporal": ("dt", "dy", "update_theta", "update_r"),
    }
    names = {
        representation: (
            *(
                f"link_{link_index}_{feature}"
                for link_index in range(VECTOR_STEPS)
                for feature in features
            ),
            "anchor_x",
        )
        for representation, features in per_link.items()
    }
    names["raw_matched"] = (
        *(f"diff_lag_{lag}" for lag in range(RAW_MATCHED_STEPS, 0, -1)),
        "anchor_x",
    )
    return names


def _validate_signal(signal: K7Signal) -> NDArray[np.float64]:
    if not isinstance(signal, K7Signal):
        msg = "signal must be a K7Signal"
        raise TypeError(msg)
    expected_latent_names = {
        "frequency_modulation": "f",
        "baseline_modulation": "mu",
        "crest_asymmetry_modulation": "kappa",
    }
    if expected_latent_names.get(signal.mechanism) != signal.latent_name:
        msg = "K7 signal mechanism and latent_name are inconsistent"
        raise ValueError(msg)
    values = np.asarray(signal.values)
    if values.ndim != 1 or values.size <= max(HORIZONS) + RAW_MATCHED_STEPS:
        msg = "K7 signal is too short for the registered origins and horizons"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        msg = "K7 signal values must be real numeric observations"
        raise TypeError(msg)
    validated = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "K7 signal values must be finite"
        raise ValueError(msg)
    for latent in (signal.latent_coordinate, signal.latent_derivative):
        latent_values = np.asarray(latent)
        if latent_values.shape != validated.shape or not np.issubdtype(
            latent_values.dtype, np.number
        ):
            msg = "K7 latent arrays must be real numeric and match signal values"
            raise ValueError(msg)
        if np.issubdtype(latent_values.dtype, np.complexfloating) or not np.all(
            np.isfinite(latent_values)
        ):
            msg = "K7 latent arrays must be finite real values"
            raise ValueError(msg)
    return validated


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        msg = "seed must be a non-negative integer"
        raise TypeError(msg)
    validated = int(seed)
    if validated < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    return validated


def _validate_n_points(n_points: int) -> int:
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, Integral):
        msg = "n_points must be an integer greater than or equal to 2"
        raise TypeError(msg)
    validated = int(n_points)
    if validated < 2:
        msg = "n_points must be greater than or equal to 2"
        raise ValueError(msg)
    return validated


def _validate_noise_std(noise_std: float) -> float:
    if isinstance(noise_std, (bool, np.bool_)) or not isinstance(noise_std, Real):
        msg = "noise_std must be a finite non-negative real number"
        raise TypeError(msg)
    validated = float(noise_std)
    if not np.isfinite(validated) or validated < 0.0:
        msg = "noise_std must be finite and non-negative"
        raise ValueError(msg)
    return validated


def _readonly_float(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    result.flags.writeable = False
    return result


def _readonly_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64)
    result.flags.writeable = False
    return result
