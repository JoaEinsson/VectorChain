"""Pre-registered causal event-state forecasting and recursive rollout benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import forecasting
import forecasting_robustness
from vectorchain import Segment, VectorChain

VECTOR_REPRESENTATIONS = (
    "vectorchain_cartesian",
    "vectorchain_absolute",
    "vectorchain_relational",
)
EVENT_MODEL_REPRESENTATIONS = (
    *VECTOR_REPRESENTATIONS,
    "raw_matched",
    "fixed_relational",
)
REPRESENTATIONS = (*EVENT_MODEL_REPRESENTATIONS, "raw_ar", "persistence")
TARGET_NAMES = ("log1p_remaining_dt", "remaining_dy", "next_open_dy")

EVENT_FIELDS = (
    "seed",
    "signal",
    "signal_seed",
    "event_index",
    "start",
    "end",
    "emitted_at",
    "start_value",
    "end_value",
    "dt",
    "dy",
    "theta",
    "r",
    "delta_theta",
    "delta_r",
    "open_dy",
)
MODEL_FIELDS = (
    "seed",
    "history_length",
    "representation",
    "fixed_segment_length",
    "n_train_examples",
    "n_input_features",
    "n_outputs",
    "n_predictive_parameters",
    "scaler_state_bytes",
    "model_state_bytes",
    "train_runtime_s",
)
EVENT_PREDICTION_FIELDS = (
    "seed",
    "history_length",
    "representation",
    "example_id",
    "signal",
    "split",
    "origin",
    "target_emitted_at",
    "actual_remaining_dt",
    "raw_predicted_remaining_dt",
    "predicted_remaining_dt",
    "duration_raw_invalid",
    "duration_was_clipped",
    "actual_remaining_dy",
    "raw_predicted_remaining_dy",
    "predicted_remaining_dy",
    "zero_duration_displacement_projected",
    "actual_next_open_dy",
    "predicted_next_open_dy",
    "actual_dt",
    "predicted_dt",
    "actual_dy",
    "predicted_dy",
    "actual_theta",
    "predicted_theta",
    "postprojection_valid",
    "input_steps",
    "input_scalar_elements",
    "input_raw_span",
    "n_predictive_parameters",
)
ROLLOUT_FIELDS = (
    "seed",
    "history_length",
    "representation",
    "example_id",
    "signal",
    "split",
    "origin",
    "horizon",
    "n_forecast_samples",
    "squared_error_sum",
    "trajectory_rmse",
    "endpoint_absolute_error",
    "n_predicted_events",
    "preprojection_invalid_states",
    "duration_clips",
    "zero_duration_displacement_projections",
    "postprojection_valid",
    "rollout_complete",
    "input_steps",
    "input_scalar_elements",
    "input_bytes",
    "input_raw_span",
    "n_predictive_parameters",
    "rollout_runtime_s",
)
CONDITION_FIELDS = (
    "seed",
    "history_length",
    "representation",
    "split",
    "horizon",
    "status",
    "n_origins",
    "n_forecast_samples",
    "trajectory_rmse",
    "endpoint_mae",
    "mean_predicted_events",
    "preprojection_invalid_state_rate",
    "duration_clip_rate",
    "zero_duration_displacement_projection_rate",
    "postprojection_valid_rate",
    "rollout_completion_rate",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "mean_input_bytes",
    "mean_input_raw_span",
    "n_predictive_parameters",
    "rollout_runtime_median_s",
    "candidate_rmse_ratio_vs_control",
)
SIGNAL_CONDITION_FIELDS = (*CONDITION_FIELDS[:5], "signal", *CONDITION_FIELDS[5:])
SUMMARY_FIELDS = (
    "history_length",
    "representation",
    "split",
    "horizon",
    "n_seeds",
    "trajectory_rmse_geomean",
    "candidate_rmse_ratio_vs_control_geomean",
    "postprojection_valid_rate_mean",
    "rollout_completion_rate_mean",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "n_predictive_parameters",
)
SEED_SUMMARY_FIELDS = (
    "control",
    "split",
    "history_length",
    "seed",
    "n_horizons",
    "geometric_mean_candidate_rmse_ratio_vs_control",
    "candidate_seed_success",
)


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Validated Stage-10A protocol layered over the forecasting signal definitions."""

    name: str
    phase: str
    base_config_path: Path
    base: forecasting.ForecastConfig
    seeds: tuple[int, ...]
    n_points: int
    tolerance: float
    min_segment_length: int
    history_lengths: tuple[int, ...]
    candidate: str
    representations: tuple[str, ...]
    feature_names: Mapping[str, tuple[str, ...]]
    target_names: tuple[str, ...]
    duration_projection: str
    maximum_remaining_dt: int
    zero_duration_displacement: str
    gate_names: tuple[str, ...]
    fixed_segment_length: str
    raw_lags_per_candidate_feature: int
    train_fraction: float
    validation_fraction: float
    minimum_examples_per_signal_split: int
    model_kind: str
    alpha: float
    raw_horizons: tuple[int, ...]
    origin_event_stride: int
    maximum_predicted_events: int
    primary_split: str
    primary_history_length: int
    maximum_candidate_rmse_ratio: float
    robust_seed_rate: float
    minimum_robust_horizon_rate: float
    required_postprojection_validity_rate: float
    required_rollout_completion_rate: float
    maximum_candidate_step_fraction_vs_raw_matched: float
    maximum_candidate_scalar_fraction_vs_raw_matched: float
    repetitions: int
    warmup_repetitions: int
    output_root: str
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EmissionState:
    """One causally observable emitted segment plus the next open increment."""

    start: int
    end: int
    emitted_at: int
    start_value: float
    end_value: float
    dt: int
    dy: float
    theta: float
    radius: float
    delta_theta: float
    delta_radius: float
    open_dy: float


@dataclass(frozen=True, slots=True)
class EventExample:
    """One next-event example with a chronological split and exact causal history."""

    example_id: str
    signal: str
    signal_seed: int
    split: str
    state_index: int
    origin: int
    history: tuple[EmissionState, ...]
    target: EmissionState
    rollout_eligible: bool


@dataclass(frozen=True, slots=True)
class MultiRidgeModel:
    """Training-standardized deterministic multioutput ridge model."""

    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]
    coefficients_: NDArray[np.float64]
    intercept_: NDArray[np.float64]

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict one or more output columns without updating scaler state."""

        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.mean_.size:
            raise ValueError("inputs have the wrong shape for this multioutput ridge")
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must be finite")
        return np.asarray(
            ((values - self.mean_) / self.scale_) @ self.coefficients_ + self.intercept_,
            dtype=np.float64,
        )

    @property
    def n_predictive_parameters(self) -> int:
        """Return coefficient and intercept count, excluding scaler state."""

        return int(self.coefficients_.size + self.intercept_.size)

    @property
    def scaler_state_bytes(self) -> int:
        """Return bytes used by learned input mean and scale."""

        return self.mean_.nbytes + self.scale_.nbytes

    @property
    def state_bytes(self) -> int:
        """Return total bytes used by learned arrays."""

        return self.scaler_state_bytes + self.coefficients_.nbytes + self.intercept_.nbytes


@dataclass(frozen=True, slots=True)
class DurationProjection:
    """Raw and projected duration output with explicit validity metadata."""

    raw_remaining_dt: float
    remaining_dt: int
    raw_invalid: bool
    was_clipped: bool


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of one Stage-10A run."""

    run_id: str
    run_dir: Path
    summary_path: Path
    gate_path: Path
    n_conditions: int
    n_failures: int
    gate_passed: bool


def load_config(path: Path) -> RolloutConfig:
    """Load and validate the pre-registered Stage-10A configuration."""

    path = path.resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = forecasting._table(raw, "experiment")
    signals = forecasting._table(raw, "signals")
    event_state = forecasting._table(raw, "event_state")
    feature_table = forecasting._table(event_state, "features")
    targets = forecasting._table(raw, "targets")
    controls = forecasting._table(raw, "controls")
    split = forecasting._table(raw, "split")
    model = forecasting._table(raw, "model")
    rollout = forecasting._table(raw, "rollout")
    criteria = forecasting._table(raw, "criteria")
    timing = forecasting._table(raw, "timing")
    output = forecasting._table(raw, "output")

    name = forecasting._non_empty_string(experiment.get("name"), "experiment.name")
    phase = forecasting._non_empty_string(experiment.get("phase"), "experiment.phase")
    base_reference = forecasting._non_empty_string(
        experiment.get("base_config"), "experiment.base_config"
    )
    base_candidate = Path(base_reference)
    base_path = base_candidate if base_candidate.is_absolute() else path.parent / base_candidate
    base_path = base_path.resolve()
    base = forecasting.load_config(base_path)
    seeds = forecasting_robustness._unique_integer_tuple(
        experiment.get("seeds"), "experiment.seeds", minimum=0
    )
    if len(seeds) < 2:
        raise ValueError("experiment.seeds must contain at least two seeds")

    n_points = forecasting._integer(signals.get("n_points"), "signals.n_points", minimum=256)
    tolerance = forecasting._real(
        event_state.get("tolerance"), "event_state.tolerance", minimum=0.0
    )
    min_segment_length = forecasting._integer(
        event_state.get("min_segment_length"),
        "event_state.min_segment_length",
        minimum=2,
    )
    history_lengths = forecasting_robustness._increasing_integer_tuple(
        event_state.get("history_lengths"), "event_state.history_lengths", minimum=2
    )
    candidate = forecasting._non_empty_string(event_state.get("candidate"), "event_state.candidate")
    representations = forecasting._unique_string_tuple(
        event_state.get("representations"), "event_state.representations"
    )
    if representations != REPRESENTATIONS:
        raise ValueError(f"event_state.representations must exactly equal {REPRESENTATIONS}")
    if candidate != "vectorchain_relational":
        raise ValueError("event_state.candidate must equal 'vectorchain_relational'")

    required_feature_keys = (*VECTOR_REPRESENTATIONS, "fixed_relational")
    if set(feature_table) != set(required_feature_keys):
        raise ValueError("event_state.features has an unexpected representation set")
    feature_names = {
        key: forecasting._unique_string_tuple(feature_table.get(key), f"event_state.features.{key}")
        for key in required_feature_keys
    }
    expected_features = {
        "vectorchain_cartesian": ("dt", "dy", "open_dy"),
        "vectorchain_absolute": ("dt", "dy", "theta", "r", "open_dy"),
        "vectorchain_relational": (
            "dt",
            "dy",
            "theta",
            "r",
            "delta_theta",
            "delta_r",
            "open_dy",
        ),
        "fixed_relational": (
            "dt",
            "dy",
            "theta",
            "r",
            "delta_theta",
            "delta_r",
            "open_dy",
        ),
    }
    if feature_names != expected_features:
        raise ValueError("event_state.features must match the registered state definitions")

    target_names = forecasting._unique_string_tuple(targets.get("names"), "targets.names")
    if target_names != TARGET_NAMES:
        raise ValueError(f"targets.names must exactly equal {TARGET_NAMES}")
    duration_projection = forecasting._non_empty_string(
        targets.get("duration_projection"), "targets.duration_projection"
    )
    if duration_projection != "round_expm1_clip":
        raise ValueError("targets.duration_projection must equal 'round_expm1_clip'")
    maximum_remaining_dt = forecasting._integer(
        targets.get("maximum_remaining_dt"), "targets.maximum_remaining_dt", minimum=1
    )
    zero_duration_displacement = forecasting._non_empty_string(
        targets.get("zero_duration_displacement"), "targets.zero_duration_displacement"
    )
    if zero_duration_displacement != "force_zero":
        raise ValueError("targets.zero_duration_displacement must equal 'force_zero'")

    gate_names = forecasting._unique_string_tuple(controls.get("gate_names"), "controls.gate_names")
    if gate_names != ("raw_matched", "fixed_relational", "raw_ar", "persistence"):
        raise ValueError("controls.gate_names must match the registered causal controls")
    fixed_segment_length = forecasting._non_empty_string(
        controls.get("fixed_segment_length"), "controls.fixed_segment_length"
    )
    if fixed_segment_length != "training_median_vectorchain_dt":
        raise ValueError("controls.fixed_segment_length has an unsupported policy")
    raw_lags_per_feature = forecasting._integer(
        controls.get("raw_lags_per_candidate_feature"),
        "controls.raw_lags_per_candidate_feature",
        minimum=1,
    )
    if raw_lags_per_feature != 1:
        raise ValueError("controls.raw_lags_per_candidate_feature must equal 1")

    train_fraction = _open_unit(split.get("train_fraction"), "split.train_fraction")
    validation_fraction = _open_unit(split.get("validation_fraction"), "split.validation_fraction")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("split fractions must leave a non-empty test partition")
    minimum_examples = forecasting._integer(
        split.get("minimum_examples_per_signal_split"),
        "split.minimum_examples_per_signal_split",
        minimum=1,
    )

    model_kind = forecasting._non_empty_string(model.get("kind"), "model.kind")
    if model_kind != "ridge":
        raise ValueError("model.kind must equal 'ridge'")
    alpha = forecasting._real(model.get("alpha"), "model.alpha", minimum=0.0)
    if alpha <= 0.0:
        raise ValueError("model.alpha must be positive")
    raw_horizons = forecasting_robustness._increasing_integer_tuple(
        rollout.get("raw_horizons"), "rollout.raw_horizons", minimum=1
    )
    origin_stride = forecasting._integer(
        rollout.get("origin_event_stride"), "rollout.origin_event_stride", minimum=1
    )
    maximum_events = forecasting._integer(
        rollout.get("maximum_predicted_events"),
        "rollout.maximum_predicted_events",
        minimum=1,
    )

    primary_split = forecasting._non_empty_string(
        criteria.get("primary_split"), "criteria.primary_split"
    )
    if primary_split != "test":
        raise ValueError("criteria.primary_split must equal 'test'")
    primary_history = forecasting._integer(
        criteria.get("primary_history_length"),
        "criteria.primary_history_length",
        minimum=2,
    )
    if primary_history not in history_lengths:
        raise ValueError("criteria.primary_history_length must be in event_state.history_lengths")
    maximum_ratio = forecasting._real(
        criteria.get("maximum_candidate_rmse_ratio"),
        "criteria.maximum_candidate_rmse_ratio",
        minimum=0.0,
    )
    if not 0.0 < maximum_ratio < 1.0:
        raise ValueError("criteria.maximum_candidate_rmse_ratio must satisfy 0 < value < 1")
    robust_seed_rate = _open_unit(criteria.get("robust_seed_rate"), "criteria.robust_seed_rate")
    robust_horizon_rate = _open_unit(
        criteria.get("minimum_robust_horizon_rate"),
        "criteria.minimum_robust_horizon_rate",
    )
    validity_rate = _open_unit(
        criteria.get("required_postprojection_validity_rate"),
        "criteria.required_postprojection_validity_rate",
    )
    completion_rate = _open_unit(
        criteria.get("required_rollout_completion_rate"),
        "criteria.required_rollout_completion_rate",
    )
    step_fraction = _positive_real(
        criteria.get("maximum_candidate_step_fraction_vs_raw_matched"),
        "criteria.maximum_candidate_step_fraction_vs_raw_matched",
    )
    scalar_fraction = _positive_real(
        criteria.get("maximum_candidate_scalar_fraction_vs_raw_matched"),
        "criteria.maximum_candidate_scalar_fraction_vs_raw_matched",
    )
    repetitions = forecasting._integer(timing.get("repetitions"), "timing.repetitions", minimum=1)
    warmups = forecasting._integer(
        timing.get("warmup_repetitions"), "timing.warmup_repetitions", minimum=0
    )
    output_root = forecasting._non_empty_string(output.get("root"), "output.root")
    save_plots = forecasting._boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = forecasting._integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    return RolloutConfig(
        name=name,
        phase=phase,
        base_config_path=base_path,
        base=base,
        seeds=seeds,
        n_points=n_points,
        tolerance=tolerance,
        min_segment_length=min_segment_length,
        history_lengths=history_lengths,
        candidate=candidate,
        representations=representations,
        feature_names=feature_names,
        target_names=target_names,
        duration_projection=duration_projection,
        maximum_remaining_dt=maximum_remaining_dt,
        zero_duration_displacement=zero_duration_displacement,
        gate_names=gate_names,
        fixed_segment_length=fixed_segment_length,
        raw_lags_per_candidate_feature=raw_lags_per_feature,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        minimum_examples_per_signal_split=minimum_examples,
        model_kind=model_kind,
        alpha=alpha,
        raw_horizons=raw_horizons,
        origin_event_stride=origin_stride,
        maximum_predicted_events=maximum_events,
        primary_split=primary_split,
        primary_history_length=primary_history,
        maximum_candidate_rmse_ratio=maximum_ratio,
        robust_seed_rate=robust_seed_rate,
        minimum_robust_horizon_rate=robust_horizon_rate,
        required_postprojection_validity_rate=validity_rate,
        required_rollout_completion_rate=completion_rate,
        maximum_candidate_step_fraction_vs_raw_matched=step_fraction,
        maximum_candidate_scalar_fraction_vs_raw_matched=scalar_fraction,
        repetitions=repetitions,
        warmup_repetitions=warmups,
        output_root=output_root,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )


def _open_unit(value: object, name: str) -> float:
    result = forecasting._real(value, name, minimum=0.0)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < value <= 1")
    return result


def _positive_real(value: object, name: str) -> float:
    result = forecasting._real(value, name, minimum=0.0)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def extract_emission_states(
    values: NDArray[np.float64], *, tolerance: float, min_segment_length: int
) -> tuple[EmissionState, ...]:
    """Extract only threshold-emitted states without finalizing the open segment."""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("values must be a finite one-dimensional signal with at least two points")
    chain = VectorChain(
        tolerance=tolerance,
        min_segment_length=min_segment_length,
        features=("dt", "dy"),
    )
    states: list[EmissionState] = []
    for index, value in enumerate(signal):
        emitted = chain.update(float(value))
        for segment in emitted:
            states.append(_observed_state(segment, signal, states[-1] if states else None))
            if segment.emitted_at != index:
                raise RuntimeError("emission timestamp diverged from the streaming clock")
    return tuple(states)


def _observed_state(
    segment: Segment,
    values: NDArray[np.float64],
    previous: EmissionState | None,
) -> EmissionState:
    if segment.emitted_at is None:
        raise ValueError("terminal finalized segments are not causal event states")
    emitted_at = int(segment.emitted_at)
    if emitted_at != segment.end + 1:
        raise RuntimeError("normal VectorChain emission must occur one step after the endpoint")
    open_dy = float(values[emitted_at] - values[emitted_at - 1])
    return _state_from_components(
        start=segment.start,
        emitted_at=emitted_at,
        start_value=segment.start_value,
        dt=segment.dt,
        dy=segment.dy,
        open_dy=open_dy,
        previous=previous,
    )


def _state_from_components(
    *,
    start: int,
    emitted_at: int,
    start_value: float,
    dt: int,
    dy: float,
    open_dy: float,
    previous: EmissionState | None,
) -> EmissionState:
    if dt < 1 or emitted_at != start + dt + 1:
        raise ValueError("an event state must have positive duration and emission=end+1")
    values = np.asarray((start_value, dy, open_dy), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("event-state values must be finite")
    theta = float(np.arctan2(dy, dt))
    radius = float(np.hypot(dt, dy))
    delta_theta = 0.0 if previous is None else theta - previous.theta
    delta_radius = 0.0 if previous is None else radius - previous.radius
    return EmissionState(
        start=start,
        end=start + dt,
        emitted_at=emitted_at,
        start_value=float(start_value),
        end_value=float(start_value + dy),
        dt=dt,
        dy=float(dy),
        theta=theta,
        radius=radius,
        delta_theta=delta_theta,
        delta_radius=delta_radius,
        open_dy=float(open_dy),
    )


def _event_rows(
    *,
    seed: int,
    signal: str,
    signal_seed: int,
    states: Sequence[EmissionState],
) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "signal": signal,
            "signal_seed": signal_seed,
            "event_index": index,
            "start": state.start,
            "end": state.end,
            "emitted_at": state.emitted_at,
            "start_value": state.start_value,
            "end_value": state.end_value,
            "dt": state.dt,
            "dy": state.dy,
            "theta": state.theta,
            "r": state.radius,
            "delta_theta": state.delta_theta,
            "delta_r": state.delta_radius,
            "open_dy": state.open_dy,
        }
        for index, state in enumerate(states)
    ]


def _split_boundaries(config: RolloutConfig) -> tuple[int, int]:
    train_end = math.floor(config.n_points * config.train_fraction)
    validation_end = math.floor(
        config.n_points * (config.train_fraction + config.validation_fraction)
    )
    return train_end, validation_end


def _split_for_index(index: int, config: RolloutConfig) -> str:
    train_end, validation_end = _split_boundaries(config)
    return "train" if index < train_end else "validation" if index < validation_end else "test"


def _training_fixed_length(
    states_by_signal: Mapping[str, Sequence[EmissionState]], config: RolloutConfig
) -> int:
    train_end, _ = _split_boundaries(config)
    durations = np.asarray(
        [
            state.dt
            for states in states_by_signal.values()
            for state in states
            if state.emitted_at < train_end
        ],
        dtype=np.float64,
    )
    if durations.size == 0:
        raise ValueError("training partition contains no emitted VectorChain segments")
    return max(1, int(np.rint(np.median(durations))))


def _build_examples(
    states_by_signal: Mapping[str, Sequence[EmissionState]],
    signal_seeds: Mapping[str, int],
    *,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> tuple[EventExample, ...]:
    examples: list[EventExample] = []
    candidate_width = len(config.feature_names[config.candidate])
    raw_lags = candidate_width * history_length * config.raw_lags_per_candidate_feature
    maximum_horizon = max(config.raw_horizons)
    for signal, states in states_by_signal.items():
        for target_index in range(history_length, len(states)):
            state_index = target_index - 1
            origin = states[state_index].emitted_at
            earliest_fixed_start = origin - 1 - history_length * fixed_length
            if origin < raw_lags or earliest_fixed_start < 0:
                continue
            target = states[target_index]
            split = _split_for_index(target.emitted_at, config)
            examples.append(
                EventExample(
                    example_id=(f"{signal}__history-{history_length}__target-event-{target_index}"),
                    signal=signal,
                    signal_seed=signal_seeds[signal],
                    split=split,
                    state_index=state_index,
                    origin=origin,
                    history=tuple(states[state_index - history_length + 1 : state_index + 1]),
                    target=target,
                    rollout_eligible=origin + maximum_horizon < config.n_points,
                )
            )
    _validate_example_counts(examples, config)
    return tuple(examples)


def _validate_example_counts(examples: Sequence[EventExample], config: RolloutConfig) -> None:
    for signal in config.base.signal_names:
        for split in ("train", "validation", "test"):
            count = sum(example.signal == signal and example.split == split for example in examples)
            if count < config.minimum_examples_per_signal_split:
                raise ValueError(
                    f"{signal}/{split} has {count} event examples; "
                    f"minimum is {config.minimum_examples_per_signal_split}"
                )
        for split in ("validation", "test"):
            eligible = sum(
                example.signal == signal and example.split == split and example.rollout_eligible
                for example in examples
            )
            if eligible < config.minimum_examples_per_signal_split:
                raise ValueError(f"{signal}/{split} has only {eligible} rollout-eligible examples")


def _target_vector(example: EventExample) -> NDArray[np.float64]:
    current = example.history[-1]
    remaining_dt = example.target.dt - 1
    remaining_dy = example.target.dy - current.open_dy
    return np.asarray(
        (np.log1p(remaining_dt), remaining_dy, example.target.open_dy), dtype=np.float64
    )


def fit_multi_ridge(
    inputs: NDArray[np.float64], targets: NDArray[np.float64], *, alpha: float
) -> MultiRidgeModel:
    """Fit deterministic multioutput ridge with training-only standardization."""

    values = np.asarray(inputs, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("inputs must be a non-empty two-dimensional matrix")
    if expected.ndim != 2 or expected.shape[0] != values.shape[0] or expected.shape[1] == 0:
        raise ValueError("targets must be a two-dimensional matrix aligned with inputs")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(expected)):
        raise ValueError("ridge inputs and targets must be finite")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (values - mean) / scale
    intercept = np.mean(expected, axis=0)
    centered = expected - intercept
    gram = standardized.T @ standardized
    penalty = alpha * np.eye(values.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram + penalty, standardized.T @ centered)
    arrays = tuple(
        np.asarray(array, dtype=np.float64) for array in (mean, scale, coefficients, intercept)
    )
    for array in arrays:
        array.flags.writeable = False
    return MultiRidgeModel(arrays[0], arrays[1], arrays[2], arrays[3])


def _state_feature_matrix(
    states: Sequence[EmissionState], feature_names: Sequence[str]
) -> NDArray[np.float64]:
    if not states:
        raise ValueError("state history must not be empty")
    dt = np.asarray([state.dt for state in states], dtype=np.float64)
    dy = np.asarray([state.dy for state in states], dtype=np.float64)
    open_dy = np.asarray([state.open_dy for state in states], dtype=np.float64)
    theta = np.arctan2(dy, dt)
    radius = np.hypot(dt, dy)
    delta_theta = np.zeros_like(theta)
    delta_radius = np.zeros_like(radius)
    if dt.size > 1:
        delta_theta[1:] = theta[1:] - theta[:-1]
        delta_radius[1:] = radius[1:] - radius[:-1]
    columns = {
        "dt": dt,
        "dy": dy,
        "theta": theta,
        "r": radius,
        "delta_theta": delta_theta,
        "delta_r": delta_radius,
        "open_dy": open_dy,
    }
    return np.column_stack(tuple(columns[name] for name in feature_names))


def _fixed_state_matrix(
    values: NDArray[np.float64],
    *,
    origin: int,
    history_length: int,
    fixed_length: int,
) -> NDArray[np.float64]:
    rows: list[tuple[float, float, float]] = []
    for offset in range(history_length - 1, -1, -1):
        end = origin - 1 - offset * fixed_length
        start = end - fixed_length
        if start < 0 or end + 1 >= values.size:
            raise ValueError("fixed-state history falls outside the observed raw prefix")
        rows.append(
            (
                float(fixed_length),
                float(values[end] - values[start]),
                float(values[end + 1] - values[end]),
            )
        )
    dt = np.asarray([row[0] for row in rows], dtype=np.float64)
    dy = np.asarray([row[1] for row in rows], dtype=np.float64)
    open_dy = np.asarray([row[2] for row in rows], dtype=np.float64)
    theta = np.arctan2(dy, dt)
    radius = np.hypot(dt, dy)
    delta_theta = np.zeros_like(theta)
    delta_radius = np.zeros_like(radius)
    if dt.size > 1:
        delta_theta[1:] = theta[1:] - theta[:-1]
        delta_radius[1:] = radius[1:] - radius[:-1]
    return np.column_stack((dt, dy, theta, radius, delta_theta, delta_radius, open_dy))


def _raw_lag_vector(
    values: NDArray[np.float64], *, origin: int, raw_lags: int
) -> NDArray[np.float64]:
    start = origin - raw_lags
    if start < 0 or origin >= values.size:
        raise ValueError("raw lag history falls outside the observed prefix")
    differences = np.diff(values[start : origin + 1])
    if differences.size != raw_lags:
        raise RuntimeError("raw lag vector has an unexpected length")
    return np.asarray(differences, dtype=np.float64)


def _input_vector(
    representation: str,
    *,
    history: Sequence[EmissionState],
    values: NDArray[np.float64],
    origin: int,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> NDArray[np.float64]:
    if len(history) < history_length:
        raise ValueError("not enough event states for the configured history")
    if representation in VECTOR_REPRESENTATIONS:
        matrix = _state_feature_matrix(
            history[-history_length:], config.feature_names[representation]
        )
        return np.asarray(matrix.reshape(-1), dtype=np.float64)
    candidate_width = len(config.feature_names[config.candidate])
    raw_lags = candidate_width * history_length * config.raw_lags_per_candidate_feature
    if representation == "raw_matched":
        return _raw_lag_vector(values, origin=origin, raw_lags=raw_lags)
    if representation == "fixed_relational":
        matrix = _fixed_state_matrix(
            values,
            origin=origin,
            history_length=history_length,
            fixed_length=fixed_length,
        )
        return np.asarray(matrix.reshape(-1), dtype=np.float64)
    raise ValueError(f"{representation} does not use the event-state multioutput model")


def project_duration(value: float, maximum_remaining_dt: int) -> DurationProjection:
    """Project one duration output with registered half-to-even rounding."""

    raw = float(np.expm1(value)) if np.isfinite(value) else float(value)
    raw_invalid = not np.isfinite(raw) or raw < 0.0
    if np.isnan(raw) or raw == -np.inf:
        safe = 0.0
    elif raw == np.inf:
        safe = float(maximum_remaining_dt)
    else:
        safe = raw
    rounded = float(np.rint(safe))
    clipped = min(float(maximum_remaining_dt), max(0.0, rounded))
    return DurationProjection(
        raw_remaining_dt=raw,
        remaining_dt=int(clipped),
        raw_invalid=raw_invalid,
        was_clipped=not np.isfinite(raw) or rounded != clipped,
    )


def project_remaining_displacement(
    value: float, duration: DurationProjection
) -> tuple[float, bool]:
    """Remove impossible displacement when no unobserved interval remains."""

    raw = float(value)
    if duration.remaining_dt == 0:
        return 0.0, raw != 0.0
    return raw, False


def _input_shape(
    representation: str,
    *,
    history_length: int,
    fixed_length: int,
    example: EventExample,
    config: RolloutConfig,
) -> tuple[int, int, int]:
    candidate_width = len(config.feature_names[config.candidate])
    if representation in VECTOR_REPRESENTATIONS:
        steps = history_length
        scalars = history_length * len(config.feature_names[representation])
        raw_span = example.origin - example.history[0].start
    elif representation == "fixed_relational":
        steps = history_length
        scalars = history_length * candidate_width
        raw_span = history_length * fixed_length + 1
    elif representation in {"raw_matched", "raw_ar"}:
        steps = history_length * candidate_width
        scalars = steps
        raw_span = steps
    else:
        steps = 1
        scalars = 1
        raw_span = 1
    return steps, scalars, raw_span


def _fit_event_models(
    examples: Sequence[EventExample],
    signals: Mapping[str, NDArray[np.float64]],
    *,
    seed: int,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> tuple[dict[str, MultiRidgeModel], list[dict[str, object]]]:
    train_examples = [example for example in examples if example.split == "train"]
    targets = np.vstack([_target_vector(example) for example in train_examples])
    models: dict[str, MultiRidgeModel] = {}
    rows: list[dict[str, object]] = []
    for representation in EVENT_MODEL_REPRESENTATIONS:
        design = np.vstack(
            [
                _input_vector(
                    representation,
                    history=example.history,
                    values=signals[example.signal],
                    origin=example.origin,
                    history_length=history_length,
                    fixed_length=fixed_length,
                    config=config,
                )
                for example in train_examples
            ]
        )
        for _ in range(config.warmup_repetitions):
            fit_multi_ridge(design, targets, alpha=config.alpha)
        durations: list[float] = []
        model: MultiRidgeModel | None = None
        for _ in range(config.repetitions):
            started = time.perf_counter_ns()
            fitted = fit_multi_ridge(design, targets, alpha=config.alpha)
            durations.append((time.perf_counter_ns() - started) / 1e9)
            if model is not None and not _models_equal(model, fitted):
                raise RuntimeError("event ridge changed across deterministic repetitions")
            model = fitted
        if model is None:
            raise RuntimeError("at least one event-model repetition is required")
        models[representation] = model
        rows.append(
            _model_row(
                seed=seed,
                history_length=history_length,
                representation=representation,
                fixed_length=fixed_length,
                n_train=design.shape[0],
                model=model,
                train_runtime_s=float(np.median(durations)),
            )
        )
    return models, rows


def _fit_raw_ar(
    signals: Mapping[str, NDArray[np.float64]],
    *,
    seed: int,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> tuple[MultiRidgeModel, dict[str, object]]:
    candidate_width = len(config.feature_names[config.candidate])
    raw_lags = candidate_width * history_length * config.raw_lags_per_candidate_feature
    train_end, _ = _split_boundaries(config)
    design_rows: list[NDArray[np.float64]] = []
    target_rows: list[tuple[float]] = []
    for values in signals.values():
        for target_index in range(raw_lags + 1, train_end):
            design_rows.append(_raw_lag_vector(values, origin=target_index - 1, raw_lags=raw_lags))
            target_rows.append((float(values[target_index] - values[target_index - 1]),))
    design = np.vstack(design_rows)
    targets = np.asarray(target_rows, dtype=np.float64)
    for _ in range(config.warmup_repetitions):
        fit_multi_ridge(design, targets, alpha=config.alpha)
    durations: list[float] = []
    model: MultiRidgeModel | None = None
    for _ in range(config.repetitions):
        started = time.perf_counter_ns()
        fitted = fit_multi_ridge(design, targets, alpha=config.alpha)
        durations.append((time.perf_counter_ns() - started) / 1e9)
        if model is not None and not _models_equal(model, fitted):
            raise RuntimeError("raw AR ridge changed across deterministic repetitions")
        model = fitted
    if model is None:
        raise RuntimeError("at least one raw-AR repetition is required")
    return model, _model_row(
        seed=seed,
        history_length=history_length,
        representation="raw_ar",
        fixed_length=fixed_length,
        n_train=design.shape[0],
        model=model,
        train_runtime_s=float(np.median(durations)),
    )


def _model_row(
    *,
    seed: int,
    history_length: int,
    representation: str,
    fixed_length: int,
    n_train: int,
    model: MultiRidgeModel,
    train_runtime_s: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "history_length": history_length,
        "representation": representation,
        "fixed_segment_length": fixed_length,
        "n_train_examples": n_train,
        "n_input_features": model.mean_.size,
        "n_outputs": model.intercept_.size,
        "n_predictive_parameters": model.n_predictive_parameters,
        "scaler_state_bytes": model.scaler_state_bytes,
        "model_state_bytes": model.state_bytes,
        "train_runtime_s": train_runtime_s,
    }


def _models_equal(left: MultiRidgeModel, right: MultiRidgeModel) -> bool:
    return all(
        np.array_equal(left_array, right_array)
        for left_array, right_array in (
            (left.mean_, right.mean_),
            (left.scale_, right.scale_),
            (left.coefficients_, right.coefficients_),
            (left.intercept_, right.intercept_),
        )
    )


def _event_prediction_rows(
    examples: Sequence[EventExample],
    signals: Mapping[str, NDArray[np.float64]],
    models: Mapping[str, MultiRidgeModel],
    *,
    seed: int,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representation, model in models.items():
        for example in examples:
            if example.split == "train":
                continue
            vector = _input_vector(
                representation,
                history=example.history,
                values=signals[example.signal],
                origin=example.origin,
                history_length=history_length,
                fixed_length=fixed_length,
                config=config,
            )
            prediction = model.predict(vector)[0]
            duration = project_duration(float(prediction[0]), config.maximum_remaining_dt)
            raw_predicted_remaining_dy = float(prediction[1])
            predicted_remaining_dy, displacement_projected = project_remaining_displacement(
                raw_predicted_remaining_dy, duration
            )
            predicted_next_open = float(prediction[2])
            current = example.history[-1]
            predicted_dt = duration.remaining_dt + 1
            predicted_dy = current.open_dy + predicted_remaining_dy
            predicted_theta = float(np.arctan2(predicted_dy, predicted_dt))
            post_valid = bool(
                np.isfinite(predicted_remaining_dy)
                and np.isfinite(predicted_next_open)
                and predicted_dt >= 1
            )
            steps, scalars, raw_span = _input_shape(
                representation,
                history_length=history_length,
                fixed_length=fixed_length,
                example=example,
                config=config,
            )
            rows.append(
                {
                    "seed": seed,
                    "history_length": history_length,
                    "representation": representation,
                    "example_id": example.example_id,
                    "signal": example.signal,
                    "split": example.split,
                    "origin": example.origin,
                    "target_emitted_at": example.target.emitted_at,
                    "actual_remaining_dt": example.target.dt - 1,
                    "raw_predicted_remaining_dt": duration.raw_remaining_dt,
                    "predicted_remaining_dt": duration.remaining_dt,
                    "duration_raw_invalid": duration.raw_invalid,
                    "duration_was_clipped": duration.was_clipped,
                    "actual_remaining_dy": example.target.dy - current.open_dy,
                    "raw_predicted_remaining_dy": raw_predicted_remaining_dy,
                    "predicted_remaining_dy": predicted_remaining_dy,
                    "zero_duration_displacement_projected": displacement_projected,
                    "actual_next_open_dy": example.target.open_dy,
                    "predicted_next_open_dy": predicted_next_open,
                    "actual_dt": example.target.dt,
                    "predicted_dt": predicted_dt,
                    "actual_dy": example.target.dy,
                    "predicted_dy": predicted_dy,
                    "actual_theta": example.target.theta,
                    "predicted_theta": predicted_theta,
                    "postprojection_valid": post_valid,
                    "input_steps": steps,
                    "input_scalar_elements": scalars,
                    "input_raw_span": raw_span,
                    "n_predictive_parameters": model.n_predictive_parameters,
                }
            )
    return rows


def _rollout_rows(
    examples: Sequence[EventExample],
    signals: Mapping[str, NDArray[np.float64]],
    event_models: Mapping[str, MultiRidgeModel],
    raw_ar_model: MultiRidgeModel,
    *,
    seed: int,
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for signal in config.base.signal_names:
        for split in ("validation", "test"):
            selected = [
                example
                for example in examples
                if example.signal == signal and example.split == split and example.rollout_eligible
            ][:: config.origin_event_stride]
            for example in selected:
                values = signals[signal]
                for representation in config.representations:
                    started = time.perf_counter_ns()
                    if representation in EVENT_MODEL_REPRESENTATIONS:
                        predicted, n_events, raw_invalid, clips, dy_projects, valid, complete = (
                            _event_model_rollout(
                                representation,
                                model=event_models[representation],
                                example=example,
                                values=values,
                                history_length=history_length,
                                fixed_length=fixed_length,
                                config=config,
                            )
                        )
                        parameters = event_models[representation].n_predictive_parameters
                    elif representation == "raw_ar":
                        predicted, valid = _raw_ar_rollout(
                            raw_ar_model,
                            values=values,
                            origin=example.origin,
                            history_length=history_length,
                            config=config,
                        )
                        n_events, raw_invalid, clips, dy_projects, complete = 0, 0, 0, 0, valid
                        parameters = raw_ar_model.n_predictive_parameters
                    else:
                        predicted = np.full(
                            max(config.raw_horizons),
                            float(values[example.origin]),
                            dtype=np.float64,
                        )
                        n_events, raw_invalid, clips, dy_projects, valid, complete, parameters = (
                            0,
                            0,
                            0,
                            0,
                            True,
                            True,
                            0,
                        )
                    runtime_s = (time.perf_counter_ns() - started) / 1e9
                    steps, scalars, raw_span = _input_shape(
                        representation,
                        history_length=history_length,
                        fixed_length=fixed_length,
                        example=example,
                        config=config,
                    )
                    for horizon in config.raw_horizons:
                        actual = values[example.origin + 1 : example.origin + horizon + 1]
                        forecast = predicted[:horizon]
                        horizon_complete = complete and forecast.size == horizon
                        if forecast.size != horizon:
                            padded = np.full(horizon, np.nan, dtype=np.float64)
                            padded[: forecast.size] = forecast
                            forecast = padded
                        errors = forecast - actual
                        squared_error_sum = (
                            float(np.sum(errors**2))
                            if np.all(np.isfinite(errors))
                            else float("nan")
                        )
                        trajectory_rmse = (
                            float(np.sqrt(squared_error_sum / horizon))
                            if np.isfinite(squared_error_sum)
                            else float("nan")
                        )
                        endpoint_error = (
                            float(abs(errors[-1])) if np.isfinite(errors[-1]) else float("nan")
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "history_length": history_length,
                                "representation": representation,
                                "example_id": example.example_id,
                                "signal": signal,
                                "split": split,
                                "origin": example.origin,
                                "horizon": horizon,
                                "n_forecast_samples": horizon,
                                "squared_error_sum": squared_error_sum,
                                "trajectory_rmse": trajectory_rmse,
                                "endpoint_absolute_error": endpoint_error,
                                "n_predicted_events": n_events,
                                "preprojection_invalid_states": raw_invalid,
                                "duration_clips": clips,
                                "zero_duration_displacement_projections": dy_projects,
                                "postprojection_valid": valid,
                                "rollout_complete": horizon_complete,
                                "input_steps": steps,
                                "input_scalar_elements": scalars,
                                "input_bytes": scalars * 8,
                                "input_raw_span": raw_span,
                                "n_predictive_parameters": parameters,
                                "rollout_runtime_s": runtime_s,
                            }
                        )
    return rows


def _event_model_rollout(
    representation: str,
    *,
    model: MultiRidgeModel,
    example: EventExample,
    values: NDArray[np.float64],
    history_length: int,
    fixed_length: int,
    config: RolloutConfig,
) -> tuple[NDArray[np.float64], int, int, int, int, bool, bool]:
    maximum_horizon = max(config.raw_horizons)
    simulated = [float(value) for value in values[: example.origin + 1]]
    state_history = list(example.history)
    generated: list[float] = []
    raw_invalid = 0
    clips = 0
    displacement_projections = 0
    valid = True
    n_events = 0
    while len(generated) < maximum_horizon and n_events < config.maximum_predicted_events:
        current_origin = len(simulated) - 1
        vector = _input_vector(
            representation,
            history=state_history,
            values=np.asarray(simulated, dtype=np.float64),
            origin=current_origin,
            history_length=history_length,
            fixed_length=fixed_length,
            config=config,
        )
        prediction = model.predict(vector)[0]
        duration = project_duration(float(prediction[0]), config.maximum_remaining_dt)
        raw_invalid += int(duration.raw_invalid)
        clips += int(duration.was_clipped)
        remaining_dy, displacement_projected = project_remaining_displacement(
            float(prediction[1]), duration
        )
        displacement_projections += int(displacement_projected)
        next_open_dy = float(prediction[2])
        if not np.isfinite(remaining_dy) or not np.isfinite(next_open_dy):
            valid = False
            break
        current_value = simulated[-1]
        if duration.remaining_dt:
            for step in range(1, duration.remaining_dt + 1):
                next_value = current_value + step * remaining_dy / duration.remaining_dt
                simulated.append(float(next_value))
                generated.append(float(next_value))
        endpoint = simulated[-1]
        next_open_value = endpoint + next_open_dy
        simulated.append(float(next_open_value))
        generated.append(float(next_open_value))

        previous = state_history[-1]
        current_open_dy = float(simulated[current_origin] - simulated[current_origin - 1])
        predicted_dt = duration.remaining_dt + 1
        predicted_state = _state_from_components(
            start=previous.end,
            emitted_at=previous.emitted_at + predicted_dt,
            start_value=previous.end_value,
            dt=predicted_dt,
            dy=current_open_dy + remaining_dy,
            open_dy=next_open_dy,
            previous=previous,
        )
        if not np.isclose(predicted_state.end_value, endpoint, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                "forward-kinematics endpoint lost continuity: "
                f"state={predicted_state.end_value}, raw={endpoint}, "
                f"start={previous.end_value}, current={current_value}, "
                f"open={current_open_dy}, remaining={remaining_dy}, "
                f"origin={current_origin}, previous_emission={previous.emitted_at}"
            )
        state_history.append(predicted_state)
        n_events += 1
    complete = len(generated) >= maximum_horizon
    return (
        np.asarray(generated[:maximum_horizon], dtype=np.float64),
        n_events,
        raw_invalid,
        clips,
        displacement_projections,
        valid,
        complete,
    )


def _raw_ar_rollout(
    model: MultiRidgeModel,
    *,
    values: NDArray[np.float64],
    origin: int,
    history_length: int,
    config: RolloutConfig,
) -> tuple[NDArray[np.float64], bool]:
    maximum_horizon = max(config.raw_horizons)
    raw_lags = len(config.feature_names[config.candidate]) * history_length
    simulated = [float(value) for value in values[: origin + 1]]
    generated: list[float] = []
    for _ in range(maximum_horizon):
        current_origin = len(simulated) - 1
        lag_vector = _raw_lag_vector(
            np.asarray(simulated, dtype=np.float64),
            origin=current_origin,
            raw_lags=raw_lags,
        )
        increment = float(model.predict(lag_vector)[0, 0])
        if not np.isfinite(increment):
            return np.asarray(generated, dtype=np.float64), False
        next_value = simulated[-1] + increment
        simulated.append(next_value)
        generated.append(next_value)
    return np.asarray(generated, dtype=np.float64), True


def _aggregate_rollouts(
    rows: Sequence[Mapping[str, object]], *, by_signal: bool
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key: tuple[object, ...] = (
            row["seed"],
            row["history_length"],
            row["representation"],
            row["split"],
            row["horizon"],
        )
        if by_signal:
            key = (*key, row["signal"])
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for key, group in groups.items():
        total_samples = sum(int(row["n_forecast_samples"]) for row in group)
        squared_errors = np.asarray([float(row["squared_error_sum"]) for row in group])
        status = "ok" if np.all(np.isfinite(squared_errors)) else "error"
        result: dict[str, object] = {
            "seed": key[0],
            "history_length": key[1],
            "representation": key[2],
            "split": key[3],
            "horizon": key[4],
        }
        if by_signal:
            result["signal"] = key[5]
        result.update(
            {
                "status": status,
                "n_origins": len(group),
                "n_forecast_samples": total_samples,
                "trajectory_rmse": (
                    float(np.sqrt(np.sum(squared_errors) / total_samples)) if status == "ok" else ""
                ),
                "endpoint_mae": _mean_finite(group, "endpoint_absolute_error"),
                "mean_predicted_events": _mean_finite(group, "n_predicted_events"),
                "preprojection_invalid_state_rate": float(
                    np.mean([int(row["preprojection_invalid_states"]) > 0 for row in group])
                ),
                "duration_clip_rate": float(
                    np.mean([int(row["duration_clips"]) > 0 for row in group])
                ),
                "zero_duration_displacement_projection_rate": float(
                    np.mean(
                        [int(row["zero_duration_displacement_projections"]) > 0 for row in group]
                    )
                ),
                "postprojection_valid_rate": float(
                    np.mean([bool(row["postprojection_valid"]) for row in group])
                ),
                "rollout_completion_rate": float(
                    np.mean([bool(row["rollout_complete"]) for row in group])
                ),
                "mean_input_steps": _mean_finite(group, "input_steps"),
                "mean_input_scalar_elements": _mean_finite(group, "input_scalar_elements"),
                "mean_input_bytes": _mean_finite(group, "input_bytes"),
                "mean_input_raw_span": _mean_finite(group, "input_raw_span"),
                "n_predictive_parameters": int(group[0]["n_predictive_parameters"]),
                "rollout_runtime_median_s": float(
                    np.median([float(row["rollout_runtime_s"]) for row in group])
                ),
                "candidate_rmse_ratio_vs_control": "",
            }
        )
        output.append(result)
    _add_candidate_ratios(output)
    return sorted(
        output,
        key=lambda row: (
            int(row["seed"]),
            int(row["history_length"]),
            str(row["split"]),
            int(row["horizon"]),
            str(row.get("signal", "")),
            str(row["representation"]),
        ),
    )


def _mean_finite(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")


def _add_candidate_ratios(rows: Sequence[dict[str, object]]) -> None:
    def comparison_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            row["seed"],
            row["history_length"],
            row["split"],
            row["horizon"],
            row.get("signal", ""),
        )

    candidates = {
        comparison_key(row): row
        for row in rows
        if row["representation"] == "vectorchain_relational" and row["status"] == "ok"
    }
    for row in rows:
        candidate = candidates.get(comparison_key(row))
        if candidate is None or row["status"] != "ok":
            continue
        control_rmse = float(row["trajectory_rmse"])
        candidate_rmse = float(candidate["trajectory_rmse"])
        if control_rmse <= 0.0:
            raise RuntimeError("rollout RMSE must be positive for paired ratios")
        row["candidate_rmse_ratio_vs_control"] = candidate_rmse / control_rmse


def _summarize_conditions(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["history_length"]),
                str(row["representation"]),
                str(row["split"]),
                int(row["horizon"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key, group in groups.items():
        successful = [row for row in group if row["status"] == "ok"]
        ratios = np.asarray(
            [float(row["candidate_rmse_ratio_vs_control"]) for row in successful],
            dtype=np.float64,
        )
        output.append(
            {
                "history_length": key[0],
                "representation": key[1],
                "split": key[2],
                "horizon": key[3],
                "n_seeds": len(successful),
                "trajectory_rmse_geomean": _geometric_mean(
                    np.asarray([float(row["trajectory_rmse"]) for row in successful])
                ),
                "candidate_rmse_ratio_vs_control_geomean": _geometric_mean(ratios),
                "postprojection_valid_rate_mean": float(
                    np.mean([float(row["postprojection_valid_rate"]) for row in successful])
                ),
                "rollout_completion_rate_mean": float(
                    np.mean([float(row["rollout_completion_rate"]) for row in successful])
                ),
                "mean_input_steps": float(
                    np.mean([float(row["mean_input_steps"]) for row in successful])
                ),
                "mean_input_scalar_elements": float(
                    np.mean([float(row["mean_input_scalar_elements"]) for row in successful])
                ),
                "n_predictive_parameters": int(successful[0]["n_predictive_parameters"]),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["history_length"]),
            str(row["split"]),
            int(row["horizon"]),
            str(row["representation"]),
        ),
    )


def _summarize_seeds(
    rows: Sequence[Mapping[str, object]], config: RolloutConfig
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for split in ("validation", "test"):
        for history_length in config.history_lengths:
            for control in config.gate_names:
                for seed in config.seeds:
                    selected = [
                        row
                        for row in rows
                        if row["representation"] == control
                        and row["split"] == split
                        and int(row["history_length"]) == history_length
                        and int(row["seed"]) == seed
                        and row["status"] == "ok"
                    ]
                    ratios = np.asarray(
                        [float(row["candidate_rmse_ratio_vs_control"]) for row in selected]
                    )
                    geomean = _geometric_mean(ratios)
                    output.append(
                        {
                            "control": control,
                            "split": split,
                            "history_length": history_length,
                            "seed": seed,
                            "n_horizons": len(selected),
                            "geometric_mean_candidate_rmse_ratio_vs_control": geomean,
                            "candidate_seed_success": (
                                geomean <= config.maximum_candidate_rmse_ratio
                            ),
                        }
                    )
    return output


def _evaluate_gate(
    conditions: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    config: RolloutConfig,
    *,
    expected_conditions: int,
) -> dict[str, object]:
    required_seeds = math.ceil(len(config.seeds) * config.robust_seed_rate)
    required_horizons = math.ceil(len(config.raw_horizons) * config.minimum_robust_horizon_rate)
    primary = [
        row
        for row in conditions
        if row["split"] == config.primary_split
        and int(row["history_length"]) == config.primary_history_length
    ]
    candidate_rows = [row for row in primary if row["representation"] == config.candidate]
    controls: dict[str, object] = {}
    for control in config.gate_names:
        control_rows = [row for row in primary if row["representation"] == control]
        selected_seeds = [
            row
            for row in seed_rows
            if row["control"] == control
            and row["split"] == config.primary_split
            and int(row["history_length"]) == config.primary_history_length
        ]
        seed_successes = sum(bool(row["candidate_seed_success"]) for row in selected_seeds)
        robust_horizons = sum(
            sum(
                float(row["candidate_rmse_ratio_vs_control"]) <= config.maximum_candidate_rmse_ratio
                for row in control_rows
                if int(row["horizon"]) == horizon
            )
            >= required_seeds
            for horizon in config.raw_horizons
        )
        checks = {
            "complete_horizon_grid": len(control_rows)
            == len(config.seeds) * len(config.raw_horizons),
            "seed_superiority": seed_successes >= required_seeds,
            "robust_horizons": robust_horizons >= required_horizons,
        }
        controls[control] = {
            "checks": checks,
            "observed": {
                "seed_successes": seed_successes,
                "seed_trials": len(selected_seeds),
                "robust_horizons": robust_horizons,
                "horizon_trials": len(config.raw_horizons),
            },
            "passed": all(checks.values()),
        }

    raw_rows = [row for row in primary if row["representation"] == "raw_matched"]
    matched = {(int(row["seed"]), int(row["horizon"])): row for row in raw_rows}
    structural_checks = {
        "step_fraction": all(
            float(row["mean_input_steps"])
            / float(matched[(int(row["seed"]), int(row["horizon"]))]["mean_input_steps"])
            <= config.maximum_candidate_step_fraction_vs_raw_matched + 1e-12
            for row in candidate_rows
        ),
        "scalar_fraction": all(
            float(row["mean_input_scalar_elements"])
            / float(matched[(int(row["seed"]), int(row["horizon"]))]["mean_input_scalar_elements"])
            <= config.maximum_candidate_scalar_fraction_vs_raw_matched + 1e-12
            for row in candidate_rows
        ),
        "matched_predictive_parameters": all(
            int(row["n_predictive_parameters"])
            == int(matched[(int(row["seed"]), int(row["horizon"]))]["n_predictive_parameters"])
            for row in candidate_rows
        ),
    }
    execution_checks = {
        "complete_condition_grid": len(conditions) == expected_conditions,
        "zero_condition_failures": all(row["status"] == "ok" for row in conditions),
        "postprojection_validity": all(
            float(row["postprojection_valid_rate"]) >= config.required_postprojection_validity_rate
            for row in candidate_rows
        ),
        "rollout_completion": all(
            float(row["rollout_completion_rate"]) >= config.required_rollout_completion_rate
            for row in candidate_rows
        ),
    }
    passed = (
        all(execution_checks.values())
        and all(structural_checks.values())
        and all(bool(value["passed"]) for value in controls.values())
    )
    return {
        "status": "evaluated",
        "hypothesis": "K5-A",
        "candidate": config.candidate,
        "primary_split": config.primary_split,
        "primary_history_length": config.primary_history_length,
        "criteria": {
            "maximum_candidate_rmse_ratio": config.maximum_candidate_rmse_ratio,
            "required_seeds": required_seeds,
            "required_horizons": required_horizons,
        },
        "execution_checks": execution_checks,
        "structural_checks": structural_checks,
        "controls": controls,
        "passed": passed,
    }


def _geometric_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0 or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(values))))


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute the complete pre-registered Stage-10A experiment."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.save_plots:
        import matplotlib

        matplotlib.use("Agg")
    repository_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = forecasting._git_state(repository_root)
    config_bytes = config_path.read_bytes()
    base_bytes = config.base_config_path.read_bytes()
    config_digest = hashlib.sha256(config_bytes + b"\0" + base_bytes).hexdigest()
    started = datetime.now(UTC)
    run_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{config_digest[:8]}_{git_commit[:7]}"
    resolved_root = (
        output_root.resolve()
        if output_root is not None
        else (repository_root / config.output_root).resolve()
    )
    run_dir = resolved_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = run_dir / "plots"
    if config.save_plots:
        plots_dir.mkdir()

    expected_conditions = (
        len(config.seeds)
        * len(config.history_lengths)
        * len(config.representations)
        * 2
        * len(config.raw_horizons)
    )
    effective = dict(config.raw)
    effective["base_forecasting_config"] = config.base.raw
    effective["resolved"] = {
        "config_path": forecasting._display_path(config_path, repository_root),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "base_config_path": forecasting._display_path(config.base_config_path, repository_root),
        "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "combined_sha256": config_digest,
        "output_root": str(resolved_root),
        "expected_condition_rows": expected_conditions,
    }
    forecasting._write_json(run_dir / "config.json", effective)
    environment = forecasting_robustness._environment_manifest(
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=effective,
        command_args=tuple(command_args if command_args is not None else sys.argv),
    )
    forecasting._write_json(run_dir / "environment.json", environment)

    event_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    event_prediction_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    derived_seeds: dict[str, dict[str, int]] = {}
    fixed_lengths: dict[str, int] = {}
    run_started = time.perf_counter_ns()
    try:
        for seed in config.seeds:
            seed_config = replace(
                config.base,
                seed=seed,
                n_points=config.n_points,
                train_fraction=config.train_fraction,
                validation_fraction=config.validation_fraction,
                vectorchain_tolerance=config.tolerance,
                vectorchain_min_segment_length=config.min_segment_length,
                save_models=False,
                save_plots=False,
                repetitions=config.repetitions,
                warmup_repetitions=config.warmup_repetitions,
            )
            signals, signal_seeds = forecasting._generate_signals(seed_config)
            derived_seeds[str(seed)] = signal_seeds
            states_by_signal = {
                signal: extract_emission_states(
                    values,
                    tolerance=config.tolerance,
                    min_segment_length=config.min_segment_length,
                )
                for signal, values in signals.items()
            }
            fixed_length = _training_fixed_length(states_by_signal, config)
            fixed_lengths[str(seed)] = fixed_length
            for signal, states in states_by_signal.items():
                event_rows.extend(
                    _event_rows(
                        seed=seed,
                        signal=signal,
                        signal_seed=signal_seeds[signal],
                        states=states,
                    )
                )

            for history_length in config.history_lengths:
                examples = _build_examples(
                    states_by_signal,
                    signal_seeds,
                    history_length=history_length,
                    fixed_length=fixed_length,
                    config=config,
                )
                event_models, fitted_rows = _fit_event_models(
                    examples,
                    signals,
                    seed=seed,
                    history_length=history_length,
                    fixed_length=fixed_length,
                    config=config,
                )
                raw_ar_model, raw_model_row = _fit_raw_ar(
                    signals,
                    seed=seed,
                    history_length=history_length,
                    fixed_length=fixed_length,
                    config=config,
                )
                model_rows.extend((*fitted_rows, raw_model_row))
                event_prediction_rows.extend(
                    _event_prediction_rows(
                        examples,
                        signals,
                        event_models,
                        seed=seed,
                        history_length=history_length,
                        fixed_length=fixed_length,
                        config=config,
                    )
                )
                rollout_rows.extend(
                    _rollout_rows(
                        examples,
                        signals,
                        event_models,
                        raw_ar_model,
                        seed=seed,
                        history_length=history_length,
                        fixed_length=fixed_length,
                        config=config,
                    )
                )
                _write_partial(
                    run_dir,
                    event_rows,
                    model_rows,
                    event_prediction_rows,
                    rollout_rows,
                )

        condition_rows = _aggregate_rollouts(rollout_rows, by_signal=False)
        signal_condition_rows = _aggregate_rollouts(rollout_rows, by_signal=True)
        summary_rows = _summarize_conditions(condition_rows)
        seed_summary_rows = _summarize_seeds(condition_rows, config)
        gate = _evaluate_gate(
            condition_rows,
            seed_summary_rows,
            config,
            expected_conditions=expected_conditions,
        )
        forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
        forecasting._write_csv(
            run_dir / "conditions_by_signal.csv",
            SIGNAL_CONDITION_FIELDS,
            signal_condition_rows,
        )
        forecasting._write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        forecasting._write_csv(
            run_dir / "summary_by_seed.csv", SEED_SUMMARY_FIELDS, seed_summary_rows
        )
        forecasting._write_json(run_dir / "gate.json", gate)
        if config.save_plots:
            _plot_summaries(
                condition_rows,
                event_prediction_rows,
                gate,
                config,
                plots_dir,
                run_id,
            )
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_started) / 1e9
        environment["derived_signal_seeds"] = derived_seeds
        environment["fixed_segment_lengths"] = fixed_lengths
        forecasting._write_json(run_dir / "environment.json", environment)
        forecasting._write_manifest(run_dir, status="failed")
        raise

    environment["status"] = "complete"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_started) / 1e9
    environment["n_conditions"] = expected_conditions
    environment["n_failures"] = 0
    environment["scientific_gate_passed"] = bool(gate["passed"])
    environment["derived_signal_seeds"] = derived_seeds
    environment["fixed_segment_lengths"] = fixed_lengths
    forecasting._write_json(run_dir / "environment.json", environment)
    forecasting._write_manifest(run_dir, status="complete")
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        summary_path=run_dir / "summary.csv",
        gate_path=run_dir / "gate.json",
        n_conditions=expected_conditions,
        n_failures=0,
        gate_passed=bool(gate["passed"]),
    )


def _write_partial(
    run_dir: Path,
    event_rows: Sequence[Mapping[str, object]],
    model_rows: Sequence[Mapping[str, object]],
    event_prediction_rows: Sequence[Mapping[str, object]],
    rollout_rows: Sequence[Mapping[str, object]],
) -> None:
    forecasting._write_csv(run_dir / "events.csv", EVENT_FIELDS, event_rows)
    forecasting._write_csv(run_dir / "models.csv", MODEL_FIELDS, model_rows)
    forecasting._write_csv(
        run_dir / "event_predictions.csv", EVENT_PREDICTION_FIELDS, event_prediction_rows
    )
    forecasting._write_csv(run_dir / "rollouts.csv", ROLLOUT_FIELDS, rollout_rows)


def _plot_summaries(
    conditions: Sequence[Mapping[str, object]],
    event_predictions: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    config: RolloutConfig,
    plots_dir: Path,
    run_id: str,
) -> None:
    import matplotlib.pyplot as plt

    primary = [
        row
        for row in conditions
        if row["split"] == config.primary_split
        and int(row["history_length"]) == config.primary_history_length
    ]
    figure, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    distributions = [
        [
            float(row["candidate_rmse_ratio_vs_control"])
            for row in primary
            if row["representation"] == control
        ]
        for control in config.gate_names
    ]
    axis.boxplot(distributions, tick_labels=config.gate_names, showmeans=True)
    axis.axhline(config.maximum_candidate_rmse_ratio, color="black", linestyle="--")
    axis.set_ylabel("RMSE(candidate) / RMSE(control)")
    axis.set_title(f"Stage-10A paired rollout ratios\nrun={run_id}", fontsize=10.0)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(plots_dir / "summary__rollout-ratio-distributions.png", dpi=config.plot_dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    positions = np.arange(len(config.raw_horizons), dtype=np.float64)
    width = 0.11
    for offset, representation in enumerate(config.representations):
        values = [
            _geometric_mean(
                np.asarray(
                    [
                        float(row["trajectory_rmse"])
                        for row in primary
                        if row["representation"] == representation
                        and int(row["horizon"]) == horizon
                    ]
                )
            )
            for horizon in config.raw_horizons
        ]
        axis.bar(
            positions + (offset - 3) * width,
            values,
            width=width,
            label=representation,
        )
    axis.set_xticks(positions, [str(value) for value in config.raw_horizons])
    axis.set_xlabel("Raw horizon")
    axis.set_ylabel("Geometric mean trajectory RMSE")
    axis.set_yscale("log")
    axis.set_title(f"Test rollout error at history 8\nrun={run_id}", fontsize=10.0)
    axis.legend(fontsize=7.5, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(plots_dir / "summary__rollout-rmse-by-horizon.png", dpi=config.plot_dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    candidate = [
        row
        for row in primary
        if row["representation"] == config.candidate
        and int(row["horizon"]) == max(config.raw_horizons)
    ]
    invalid = [float(row["preprojection_invalid_state_rate"]) for row in candidate]
    clipped = [float(row["duration_clip_rate"]) for row in candidate]
    complete = [float(row["rollout_completion_rate"]) for row in candidate]
    axis.boxplot(
        (invalid, clipped, complete),
        tick_labels=("raw invalid", "duration clipped", "complete"),
        showmeans=True,
    )
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Rate by seed")
    axis.set_title(f"Candidate validity and completion\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(plots_dir / "summary__rollout-validity.png", dpi=config.plot_dpi)
    plt.close(figure)

    selected_predictions = [
        row
        for row in event_predictions
        if row["representation"] == config.candidate
        and row["split"] == config.primary_split
        and int(row["history_length"]) == config.primary_history_length
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    axes[0].scatter(
        [float(row["actual_remaining_dt"]) for row in selected_predictions],
        [float(row["predicted_remaining_dt"]) for row in selected_predictions],
        alpha=0.25,
        s=12,
    )
    duration_limit = max(
        max(float(row["actual_remaining_dt"]) for row in selected_predictions), 1.0
    )
    axes[0].plot((0, duration_limit), (0, duration_limit), color="black", linestyle="--")
    axes[0].set_xlabel("Actual remaining dt")
    axes[0].set_ylabel("Projected predicted remaining dt")
    axes[1].scatter(
        [float(row["actual_remaining_dy"]) for row in selected_predictions],
        [float(row["predicted_remaining_dy"]) for row in selected_predictions],
        alpha=0.25,
        s=12,
    )
    axes[1].set_xlabel("Actual remaining dy")
    axes[1].set_ylabel("Predicted remaining dy")
    figure.suptitle(f"Next-event targets\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__next-event-predictions.png", dpi=config.plot_dpi)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/vector_state_rollout.toml"),
        help="TOML Stage-10A vector-state rollout configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line Stage-10A benchmark."""

    arguments = _build_parser().parse_args(argv)
    result = run_experiment(
        arguments.config,
        output_root=arguments.output_root,
        command_args=tuple(sys.argv if argv is None else (Path(__file__).name, *argv)),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "n_conditions": result.n_conditions,
                "n_failures": result.n_failures,
                "gate_passed": result.gate_passed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
