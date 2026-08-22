"""Reproducible minimal forecasting benchmark with shared downstream ridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vectorchain import (
    VectorChain,
    first_difference,
    generate_chirp,
    generate_first_order_response,
    generate_piecewise_linear,
    generate_ramp,
    generate_regime_change,
    generate_second_order_response,
    generate_sine,
    raw_values,
)
from vectorchain.features import FeatureName, validate_feature_names

SignalGenerator = Callable[..., NDArray[np.float64]]

SIGNAL_GENERATORS: dict[str, SignalGenerator] = {
    "sine": generate_sine,
    "chirp": generate_chirp,
    "ramp": generate_ramp,
    "piecewise_linear": generate_piecewise_linear,
    "first_order_response": generate_first_order_response,
    "second_order_response": generate_second_order_response,
    "regime_change": generate_regime_change,
}

SIGNAL_PARAMETERS: dict[str, frozenset[str]] = {
    "sine": frozenset({"amplitude", "offset", "frequency", "phase"}),
    "chirp": frozenset({"amplitude", "offset", "start_frequency", "end_frequency", "phase"}),
    "ramp": frozenset({"amplitude", "offset"}),
    "piecewise_linear": frozenset({"amplitude", "offset"}),
    "first_order_response": frozenset({"amplitude", "offset", "time_constant"}),
    "second_order_response": frozenset(
        {"amplitude", "offset", "natural_frequency", "damping_ratio"}
    ),
    "regime_change": frozenset(
        {
            "amplitude",
            "offset",
            "frequency_before",
            "frequency_after",
            "change_fraction",
            "level_shift",
        }
    ),
}

REPRESENTATION_NAMES = ("raw", "first_difference", "vectorchain")
SUMMARY_STATISTICS = ("last", "mean", "std")

METRIC_FIELDS = (
    "representation",
    "split",
    "status",
    "error_type",
    "error_message",
    "n_examples",
    "n_representation_features",
    "n_pooled_features",
    "n_model_parameters",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "mean_input_bytes",
    "step_reduction_factor_vs_raw",
    "scalar_reduction_factor_vs_raw",
    "mae",
    "rmse",
    "rmse_ratio_vs_raw",
    "predictive_parity_vs_raw",
    "structural_reduction_success",
    "payload_reduction_success",
    "joint_success",
    "representation_runtime_s",
    "train_runtime_median_s",
    "train_runtime_q1_s",
    "train_runtime_q3_s",
    "inference_runtime_median_s",
    "inference_runtime_q1_s",
    "inference_runtime_q3_s",
    "train_design_bytes",
    "model_state_bytes",
)

SIGNAL_METRIC_FIELDS = (
    "representation",
    "split",
    "signal",
    "n_examples",
    "mae",
    "rmse",
)

EXAMPLE_FIELDS = (
    "example_id",
    "signal",
    "signal_seed",
    "split",
    "context_start",
    "origin",
    "target_index",
    "current_value",
    "target_value",
    "target_delta",
)

INPUT_FIELDS = (
    "representation",
    "example_id",
    "signal",
    "split",
    "input_steps",
    "input_features",
    "input_scalar_elements",
    "input_bytes",
)

PREDICTION_FIELDS = (
    "representation",
    "example_id",
    "signal",
    "split",
    "origin",
    "target_index",
    "actual_delta",
    "predicted_delta",
    "actual_value",
    "predicted_value",
    "error",
)

NAIVE_FIELDS = ("split", "signal", "n_examples", "mae", "rmse")
TIMING_FIELDS = ("representation", "phase", "split", "repetition", "duration_s")


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Validated effective inputs for one minimal forecasting run."""

    name: str
    seed: int
    signal_names: tuple[str, ...]
    n_points: int
    noise_std: float
    signal_parameters: Mapping[str, Mapping[str, float]]
    context_length: int
    horizon: int
    stride: int
    train_fraction: float
    validation_fraction: float
    representation_names: tuple[str, ...]
    summary_statistics: tuple[str, ...]
    vectorchain_causal: bool
    vectorchain_tolerance: float
    vectorchain_min_segment_length: int
    vectorchain_features: tuple[FeatureName, ...]
    model_kind: str
    alpha: float
    repetitions: int
    warmup_repetitions: int
    output_root: str
    save_models: bool
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ForecastExample:
    """One causal rolling-origin supervised example."""

    example_id: str
    signal: str
    signal_seed: int
    split: str
    context_start: int
    origin: int
    target_index: int
    current_value: float
    target_value: float
    context: NDArray[np.float64]

    @property
    def target_delta(self) -> float:
        """Return the common increment target."""

        return self.target_value - self.current_value


@dataclass(frozen=True, slots=True)
class RidgeModel:
    """Training-standardized ridge model with an unregularized intercept."""

    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]
    coefficients_: NDArray[np.float64]
    intercept_: float

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict target increments without updating training statistics."""

        values = _validate_design(inputs, expected_features=self.mean_.size)
        standardized = (values - self.mean_) / self.scale_
        return np.asarray(standardized @ self.coefficients_ + self.intercept_, dtype=np.float64)

    @property
    def state_bytes(self) -> int:
        """Return bytes held by learned arrays and the scalar intercept."""

        return self.mean_.nbytes + self.scale_.nbytes + self.coefficients_.nbytes + 8


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of a completed forecasting run."""

    run_id: str
    run_dir: Path
    metrics_path: Path
    n_conditions: int
    n_failures: int


def load_config(path: Path) -> ForecastConfig:
    """Load and validate a forecasting TOML configuration."""

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    experiment = _table(raw, "experiment")
    signals = _table(raw, "signals")
    forecast = _table(raw, "forecast")
    split = _table(raw, "split")
    representations = _table(raw, "representations")
    vectorchain = _table(raw, "vectorchain")
    model = _table(raw, "model")
    output = _table(raw, "output")

    name = _non_empty_string(experiment.get("name"), "experiment.name")
    seed = _integer(experiment.get("seed"), "experiment.seed", minimum=0)
    signal_names = _unique_string_tuple(signals.get("names"), "signals.names")
    unknown = tuple(signal for signal in signal_names if signal not in SIGNAL_GENERATORS)
    if unknown:
        msg = f"unsupported signals: {unknown}"
        raise ValueError(msg)
    n_points = _integer(signals.get("n_points"), "signals.n_points", minimum=4)
    noise_std = _real(signals.get("noise_std"), "signals.noise_std", minimum=0.0)
    signal_parameters = _signal_parameter_tables(signals.get("parameters"), signal_names)

    context_length = _integer(forecast.get("context_length"), "forecast.context_length", minimum=3)
    horizon = _integer(forecast.get("horizon"), "forecast.horizon", minimum=1)
    stride = _integer(forecast.get("stride"), "forecast.stride", minimum=1)
    if context_length + horizon > n_points:
        msg = "forecast context and horizon must leave at least one supervised example"
        raise ValueError(msg)

    train_fraction = _real(split.get("train_fraction"), "split.train_fraction", minimum=0.0)
    validation_fraction = _real(
        split.get("validation_fraction"), "split.validation_fraction", minimum=0.0
    )
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        msg = "split fractions must be positive"
        raise ValueError(msg)
    if train_fraction + validation_fraction >= 1.0:
        msg = "train_fraction + validation_fraction must be less than 1"
        raise ValueError(msg)

    representation_names = _unique_string_tuple(
        representations.get("names"), "representations.names"
    )
    if representation_names != REPRESENTATION_NAMES:
        msg = f"representations.names must exactly equal {REPRESENTATION_NAMES}"
        raise ValueError(msg)
    summary_statistics = _unique_string_tuple(
        representations.get("summary_statistics"), "representations.summary_statistics"
    )
    if summary_statistics != SUMMARY_STATISTICS:
        msg = f"representations.summary_statistics must exactly equal {SUMMARY_STATISTICS}"
        raise ValueError(msg)

    vectorchain_causal = _boolean(vectorchain.get("causal"), "vectorchain.causal")
    if not vectorchain_causal:
        msg = "forecasting requires vectorchain.causal=true"
        raise ValueError(msg)
    vectorchain_tolerance = _real(
        vectorchain.get("tolerance"), "vectorchain.tolerance", minimum=0.0
    )
    vectorchain_min_segment_length = _integer(
        vectorchain.get("min_segment_length"), "vectorchain.min_segment_length", minimum=2
    )
    vectorchain_features = validate_feature_names(
        _unique_string_tuple(vectorchain.get("features"), "vectorchain.features")
    )
    VectorChain(
        tolerance=vectorchain_tolerance,
        causal=vectorchain_causal,
        min_segment_length=vectorchain_min_segment_length,
        features=vectorchain_features,
    )

    model_kind = _non_empty_string(model.get("kind"), "model.kind")
    if model_kind != "ridge":
        msg = "model.kind must equal 'ridge'"
        raise ValueError(msg)
    alpha = _real(model.get("alpha"), "model.alpha", minimum=0.0)
    if alpha <= 0.0:
        msg = "model.alpha must be positive"
        raise ValueError(msg)
    repetitions = _integer(model.get("repetitions"), "model.repetitions", minimum=1)
    warmup_repetitions = _integer(
        model.get("warmup_repetitions"), "model.warmup_repetitions", minimum=0
    )

    output_root = _non_empty_string(output.get("root"), "output.root")
    save_models = _boolean(output.get("save_models"), "output.save_models")
    save_plots = _boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = _integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    config = ForecastConfig(
        name=name,
        seed=seed,
        signal_names=signal_names,
        n_points=n_points,
        noise_std=noise_std,
        signal_parameters=signal_parameters,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        representation_names=representation_names,
        summary_statistics=summary_statistics,
        vectorchain_causal=vectorchain_causal,
        vectorchain_tolerance=vectorchain_tolerance,
        vectorchain_min_segment_length=vectorchain_min_segment_length,
        vectorchain_features=vectorchain_features,
        model_kind=model_kind,
        alpha=alpha,
        repetitions=repetitions,
        warmup_repetitions=warmup_repetitions,
        output_root=output_root,
        save_models=save_models,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )
    _validate_split_counts(config)
    return config


def fit_ridge(
    inputs: NDArray[np.float64], targets: NDArray[np.float64], *, alpha: float
) -> RidgeModel:
    """Fit deterministic ridge after training-only population standardization."""

    values = _validate_design(inputs)
    expected = np.asarray(targets)
    if expected.ndim != 1 or expected.size != values.shape[0]:
        msg = "targets must be one-dimensional with one value per input row"
        raise ValueError(msg)
    if not np.issubdtype(expected.dtype, np.number) or np.issubdtype(
        expected.dtype, np.complexfloating
    ):
        msg = "targets must contain real numeric values"
        raise TypeError(msg)
    target_values = expected.astype(np.float64, copy=False)
    if not np.all(np.isfinite(target_values)):
        msg = "targets must contain only finite values"
        raise ValueError(msg)
    validated_alpha = _real(alpha, "alpha", minimum=0.0)
    if validated_alpha <= 0.0:
        msg = "alpha must be positive"
        raise ValueError(msg)

    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (values - mean) / scale
    intercept = float(np.mean(target_values))
    centered_targets = target_values - intercept
    gram = standardized.T @ standardized
    penalty = validated_alpha * np.eye(values.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(gram + penalty, standardized.T @ centered_targets)
    arrays = tuple(np.asarray(array, dtype=np.float64) for array in (mean, scale, coefficients))
    for array in arrays:
        array.flags.writeable = False
    return RidgeModel(arrays[0], arrays[1], arrays[2], intercept)


def summarize_sequence(
    sequence: NDArray[np.float64], statistics: Sequence[str] = SUMMARY_STATISTICS
) -> NDArray[np.float64]:
    """Pool a non-empty feature sequence with registered per-column statistics."""

    values = np.asarray(sequence)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        msg = "sequence must be a non-empty one- or two-dimensional array"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        msg = "sequence must contain real numeric values"
        raise TypeError(msg)
    validated = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "sequence must contain only finite values"
        raise ValueError(msg)
    names = tuple(statistics)
    if names != SUMMARY_STATISTICS:
        msg = f"statistics must exactly equal {SUMMARY_STATISTICS}"
        raise ValueError(msg)
    return np.concatenate((validated[-1], np.mean(validated, axis=0), np.std(validated, axis=0)))


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute all representations on one shared temporal forecasting split."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.save_plots:
        import matplotlib

        matplotlib.use("Agg")

    repository_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = _git_state(repository_root)
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
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

    signals, signal_seeds = _generate_signals(config)
    examples = _build_examples(signals, signal_seeds, config)
    split_boundaries = _split_boundaries(config)
    effective_config = dict(config.raw)
    effective_config["resolved"] = {
        "config_path": _display_path(config_path, repository_root),
        "config_sha256": config_digest,
        "output_root": str(resolved_root),
        "signal_seeds": signal_seeds,
        "train_end_exclusive": split_boundaries[0],
        "validation_end_exclusive": split_boundaries[1],
        "n_examples_by_split": {
            split: sum(example.split == split for example in examples)
            for split in ("train", "validation", "test")
        },
    }
    _write_json(run_dir / "config.json", effective_config)
    environment = _environment_manifest(
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=effective_config,
        signal_seeds=signal_seeds,
        command_args=tuple(command_args if command_args is not None else sys.argv),
    )
    _write_json(run_dir / "environment.json", environment)
    _write_csv(run_dir / "examples.csv", EXAMPLE_FIELDS, _example_rows(examples))

    naive_rows = _naive_metrics(examples, config.signal_names)
    _write_csv(run_dir / "naive_metrics.csv", NAIVE_FIELDS, naive_rows)

    metric_rows: list[dict[str, object]] = []
    signal_metric_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    models: dict[str, RidgeModel] = {}
    run_start = time.perf_counter_ns()
    try:
        for representation in config.representation_names:
            try:
                result = _run_representation(representation, examples, config)
                metric_rows.extend(result[0])
                signal_metric_rows.extend(result[1])
                input_rows.extend(result[2])
                prediction_rows.extend(result[3])
                timing_rows.extend(result[4])
                models[representation] = result[5]
            except Exception as error:
                metric_rows.extend(
                    _failure_row(representation, split, error) for split in ("validation", "test")
                )
            _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metric_rows)
            _write_csv(run_dir / "metrics_by_signal.csv", SIGNAL_METRIC_FIELDS, signal_metric_rows)
            _write_csv(run_dir / "inputs.csv", INPUT_FIELDS, input_rows)
            _write_csv(run_dir / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
            _write_csv(run_dir / "timings.csv", TIMING_FIELDS, timing_rows)

        _add_raw_comparisons(metric_rows)
        _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metric_rows)
        if config.save_models and models:
            _save_models(run_dir / "models.npz", models)
        if config.save_plots:
            _plot_summaries(
                metric_rows,
                signal_metric_rows,
                naive_rows,
                plots_dir,
                run_id,
                config.plot_dpi,
            )
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
        _write_json(run_dir / "environment.json", environment)
        _write_manifest(run_dir, status="failed")
        raise

    failed_representations = {
        str(row["representation"]) for row in metric_rows if row["status"] != "ok"
    }
    environment["status"] = "complete" if not failed_representations else "complete_with_failures"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
    environment["n_conditions"] = len(config.representation_names)
    environment["n_failures"] = len(failed_representations)
    _write_json(run_dir / "environment.json", environment)
    _write_manifest(run_dir, status=str(environment["status"]))
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.csv",
        n_conditions=len(config.representation_names),
        n_failures=len(failed_representations),
    )


def _run_representation(
    representation: str,
    examples: Sequence[ForecastExample],
    config: ForecastConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    RidgeModel,
]:
    representation_start = time.perf_counter_ns()
    summaries: list[NDArray[np.float64]] = []
    input_steps: list[int] = []
    input_features: list[int] = []
    input_rows: list[dict[str, object]] = []
    for example in examples:
        sequence = _transform(representation, example.context, config)
        summary = summarize_sequence(sequence, config.summary_statistics)
        steps, features = sequence.shape
        summaries.append(summary)
        input_steps.append(steps)
        input_features.append(features)
        input_rows.append(
            {
                "representation": representation,
                "example_id": example.example_id,
                "signal": example.signal,
                "split": example.split,
                "input_steps": steps,
                "input_features": features,
                "input_scalar_elements": sequence.size,
                "input_bytes": sequence.nbytes,
            }
        )
    design = np.vstack(summaries)
    representation_runtime_s = (time.perf_counter_ns() - representation_start) / 1e9
    targets = np.asarray([example.target_delta for example in examples], dtype=np.float64)
    splits = np.asarray([example.split for example in examples])
    train_mask = splits == "train"

    for _ in range(config.warmup_repetitions):
        fit_ridge(design[train_mask], targets[train_mask], alpha=config.alpha)
    train_durations: list[float] = []
    model: RidgeModel | None = None
    for _repetition in range(config.repetitions):
        started = time.perf_counter_ns()
        candidate = fit_ridge(design[train_mask], targets[train_mask], alpha=config.alpha)
        duration = (time.perf_counter_ns() - started) / 1e9
        if model is not None and not _models_equal(model, candidate):
            msg = "deterministic ridge outputs changed between timing repetitions"
            raise RuntimeError(msg)
        model = candidate
        train_durations.append(duration)
    if model is None:
        msg = "at least one training repetition is required"
        raise RuntimeError(msg)
    train_q1, train_median, train_q3 = _quartiles(train_durations)

    timing_rows = [
        {
            "representation": representation,
            "phase": "train",
            "split": "train",
            "repetition": repetition,
            "duration_s": duration,
        }
        for repetition, duration in enumerate(train_durations)
    ]
    metric_rows: list[dict[str, object]] = []
    signal_metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        indices = np.flatnonzero(splits == split)
        split_design = design[indices]
        expected_predictions: NDArray[np.float64] | None = None
        inference_durations: list[float] = []
        for repetition in range(config.repetitions):
            started = time.perf_counter_ns()
            predicted_deltas = model.predict(split_design)
            duration = (time.perf_counter_ns() - started) / 1e9
            if expected_predictions is not None and not np.array_equal(
                expected_predictions, predicted_deltas
            ):
                msg = "deterministic predictions changed between timing repetitions"
                raise RuntimeError(msg)
            expected_predictions = predicted_deltas
            inference_durations.append(duration)
            timing_rows.append(
                {
                    "representation": representation,
                    "phase": "inference",
                    "split": split,
                    "repetition": repetition,
                    "duration_s": duration,
                }
            )
        if expected_predictions is None:
            msg = "at least one inference repetition is required"
            raise RuntimeError(msg)
        inference_q1, inference_median, inference_q3 = _quartiles(inference_durations)
        selected_examples = [examples[int(index)] for index in indices]
        actual_values = np.asarray(
            [example.target_value for example in selected_examples], dtype=np.float64
        )
        predicted_values = (
            np.asarray([example.current_value for example in selected_examples], dtype=np.float64)
            + expected_predictions
        )
        errors = predicted_values - actual_values
        selected_steps = np.asarray([input_steps[int(index)] for index in indices])
        selected_features = np.asarray([input_features[int(index)] for index in indices])
        selected_elements = selected_steps * selected_features
        selected_bytes = selected_elements * np.dtype(np.float64).itemsize
        metric_rows.append(
            {
                "representation": representation,
                "split": split,
                "status": "ok",
                "error_type": "",
                "error_message": "",
                "n_examples": indices.size,
                "n_representation_features": int(selected_features[0]),
                "n_pooled_features": design.shape[1],
                "n_model_parameters": design.shape[1] + 1,
                "mean_input_steps": float(np.mean(selected_steps)),
                "mean_input_scalar_elements": float(np.mean(selected_elements)),
                "mean_input_bytes": float(np.mean(selected_bytes)),
                "step_reduction_factor_vs_raw": "",
                "scalar_reduction_factor_vs_raw": "",
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "rmse_ratio_vs_raw": "",
                "predictive_parity_vs_raw": "",
                "structural_reduction_success": "",
                "payload_reduction_success": "",
                "joint_success": "",
                "representation_runtime_s": representation_runtime_s,
                "train_runtime_median_s": train_median,
                "train_runtime_q1_s": train_q1,
                "train_runtime_q3_s": train_q3,
                "inference_runtime_median_s": inference_median,
                "inference_runtime_q1_s": inference_q1,
                "inference_runtime_q3_s": inference_q3,
                "train_design_bytes": int(np.count_nonzero(train_mask) * design.shape[1] * 8),
                "model_state_bytes": model.state_bytes,
            }
        )
        for signal_name in config.signal_names:
            signal_mask = np.asarray(
                [example.signal == signal_name for example in selected_examples], dtype=bool
            )
            signal_errors = errors[signal_mask]
            signal_metric_rows.append(
                {
                    "representation": representation,
                    "split": split,
                    "signal": signal_name,
                    "n_examples": int(np.count_nonzero(signal_mask)),
                    "mae": float(np.mean(np.abs(signal_errors))),
                    "rmse": float(np.sqrt(np.mean(signal_errors**2))),
                }
            )
        prediction_rows.extend(
            {
                "representation": representation,
                "example_id": example.example_id,
                "signal": example.signal,
                "split": split,
                "origin": example.origin,
                "target_index": example.target_index,
                "actual_delta": example.target_delta,
                "predicted_delta": predicted_delta,
                "actual_value": example.target_value,
                "predicted_value": predicted_value,
                "error": predicted_value - example.target_value,
            }
            for example, predicted_delta, predicted_value in zip(
                selected_examples, expected_predictions, predicted_values, strict=True
            )
        )
    return metric_rows, signal_metric_rows, input_rows, prediction_rows, timing_rows, model


def _transform(
    representation: str, context: NDArray[np.float64], config: ForecastConfig
) -> NDArray[np.float64]:
    if representation == "raw":
        return raw_values(context)
    if representation == "first_difference":
        return first_difference(context)
    if representation == "vectorchain":
        chain = VectorChain(
            tolerance=config.vectorchain_tolerance,
            causal=config.vectorchain_causal,
            min_segment_length=config.vectorchain_min_segment_length,
            features=config.vectorchain_features,
        )
        chain.reset()
        for value in context:
            chain.update(float(value))
        chain.finalize()
        return chain.vectors_.copy()
    msg = f"unsupported representation: {representation}"
    raise ValueError(msg)


def _generate_signals(
    config: ForecastConfig,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, int]]:
    seeds = {name: _derive_seed(config.seed, name) for name in config.signal_names}
    signals: dict[str, NDArray[np.float64]] = {}
    for name in config.signal_names:
        values = SIGNAL_GENERATORS[name](
            rng=seeds[name],
            n_points=config.n_points,
            noise_std=config.noise_std,
            **config.signal_parameters[name],
        )
        values.flags.writeable = False
        signals[name] = values
    return signals, seeds


def _build_examples(
    signals: Mapping[str, NDArray[np.float64]],
    signal_seeds: Mapping[str, int],
    config: ForecastConfig,
) -> tuple[ForecastExample, ...]:
    train_end, validation_end = _split_boundaries(config)
    first_target = config.context_length + config.horizon - 1
    examples: list[ForecastExample] = []
    for signal_name in config.signal_names:
        values = signals[signal_name]
        for target_index in range(first_target, config.n_points, config.stride):
            origin = target_index - config.horizon
            context_start = origin - config.context_length + 1
            split = (
                "train"
                if target_index < train_end
                else "validation"
                if target_index < validation_end
                else "test"
            )
            context = values[context_start : origin + 1]
            if context.size != config.context_length:
                msg = "every forecasting context must have the configured length"
                raise RuntimeError(msg)
            examples.append(
                ForecastExample(
                    example_id=f"{signal_name}__target-{target_index}",
                    signal=signal_name,
                    signal_seed=signal_seeds[signal_name],
                    split=split,
                    context_start=context_start,
                    origin=origin,
                    target_index=target_index,
                    current_value=float(values[origin]),
                    target_value=float(values[target_index]),
                    context=context,
                )
            )
    return tuple(examples)


def _example_rows(examples: Sequence[ForecastExample]) -> list[dict[str, object]]:
    return [
        {
            "example_id": example.example_id,
            "signal": example.signal,
            "signal_seed": example.signal_seed,
            "split": example.split,
            "context_start": example.context_start,
            "origin": example.origin,
            "target_index": example.target_index,
            "current_value": example.current_value,
            "target_value": example.target_value,
            "target_delta": example.target_delta,
        }
        for example in examples
    ]


def _naive_metrics(
    examples: Sequence[ForecastExample], signal_names: Sequence[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("validation", "test"):
        split_examples = [example for example in examples if example.split == split]
        for signal_name in ("__all__", *signal_names):
            selected = (
                split_examples
                if signal_name == "__all__"
                else [example for example in split_examples if example.signal == signal_name]
            )
            errors = np.asarray(
                [example.current_value - example.target_value for example in selected],
                dtype=np.float64,
            )
            rows.append(
                {
                    "split": split,
                    "signal": signal_name,
                    "n_examples": errors.size,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                }
            )
    return rows


def _add_raw_comparisons(rows: Sequence[dict[str, object]]) -> None:
    raw_by_split = {
        str(row["split"]): row
        for row in rows
        if row["representation"] == "raw" and row["status"] == "ok"
    }
    for row in rows:
        if row["status"] != "ok" or str(row["split"]) not in raw_by_split:
            continue
        raw = raw_by_split[str(row["split"])]
        step_factor = float(raw["mean_input_steps"]) / float(row["mean_input_steps"])
        scalar_factor = float(raw["mean_input_scalar_elements"]) / float(
            row["mean_input_scalar_elements"]
        )
        rmse_ratio = float(row["rmse"]) / float(raw["rmse"])
        row["step_reduction_factor_vs_raw"] = step_factor
        row["scalar_reduction_factor_vs_raw"] = scalar_factor
        row["rmse_ratio_vs_raw"] = rmse_ratio
        if row["representation"] == "vectorchain":
            predictive = rmse_ratio <= 1.10
            structural = float(row["mean_input_steps"]) <= 0.5 * float(raw["mean_input_steps"])
            payload = float(row["mean_input_scalar_elements"]) <= float(
                raw["mean_input_scalar_elements"]
            )
            row["predictive_parity_vs_raw"] = predictive
            row["structural_reduction_success"] = structural
            row["payload_reduction_success"] = payload
            row["joint_success"] = predictive and structural and payload


def _failure_row(representation: str, split: str, error: Exception) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "representation": representation,
            "split": split,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    return row


def _save_models(path: Path, models: Mapping[str, RidgeModel]) -> None:
    arrays: dict[str, NDArray[np.float64]] = {}
    for name, model in models.items():
        arrays[f"{name}__mean"] = model.mean_
        arrays[f"{name}__scale"] = model.scale_
        arrays[f"{name}__coefficients"] = model.coefficients_
        arrays[f"{name}__intercept"] = np.asarray([model.intercept_], dtype=np.float64)
    np.savez_compressed(path, **arrays)


def _plot_summaries(
    metric_rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
    naive_rows: Sequence[Mapping[str, object]],
    plots_dir: Path,
    run_id: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    test_rows = [row for row in metric_rows if row["split"] == "test" and row["status"] == "ok"]
    labels = [_display_representation(str(row["representation"])) for row in test_rows]
    positions = np.arange(len(test_rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [float(row["mae"]) for row in test_rows],
        width=width,
        label="MAE",
    )
    axis.bar(
        positions + width / 2,
        [float(row["rmse"]) for row in test_rows],
        width=width,
        label="RMSE",
    )
    naive_test = next(
        row for row in naive_rows if row["split"] == "test" and row["signal"] == "__all__"
    )
    axis.axhline(float(naive_test["rmse"]), color="black", linestyle="--", label="Persistence RMSE")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Error in original signal units")
    axis.set_title(f"One-step forecasting error\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__test-error.png", dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
    axes[0].bar(labels, [float(row["mean_input_steps"]) for row in test_rows])
    axes[0].set_ylabel("Mean sequence steps")
    axes[0].set_title("Structural length")
    axes[1].bar(labels, [float(row["mean_input_scalar_elements"]) for row in test_rows])
    axes[1].set_ylabel("Mean float64 elements")
    axes[1].set_title("Representation payload")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Forecast input size\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__input-size.png", dpi=dpi)
    plt.close(figure)

    test_signal_rows = [row for row in signal_rows if row["split"] == "test"]
    signals = tuple(dict.fromkeys(str(row["signal"]) for row in test_signal_rows))
    figure, axis = plt.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    group_width = 0.8 / max(len(test_rows), 1)
    signal_positions = np.arange(len(signals))
    for representation_index, row in enumerate(test_rows):
        representation = str(row["representation"])
        values = [
            float(
                next(
                    item["rmse"]
                    for item in test_signal_rows
                    if item["representation"] == representation and item["signal"] == signal
                )
            )
            for signal in signals
        ]
        offset = (representation_index - (len(test_rows) - 1) / 2) * group_width
        axis.bar(
            signal_positions + offset,
            values,
            width=group_width,
            label=_display_representation(representation),
        )
    axis.set_xticks(signal_positions, [_short_signal(signal) for signal in signals], rotation=20)
    axis.set_ylabel("Test RMSE")
    axis.set_title(f"Test RMSE by signal\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__test-rmse-by-signal.png", dpi=dpi)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for row, label in zip(test_rows, labels, strict=True):
        axis.scatter(
            float(row["mean_input_scalar_elements"]),
            float(row["rmse"]),
            s=70,
            label=label,
        )
    axis.set_xlabel("Mean float64 elements before pooling")
    axis.set_ylabel("Test RMSE")
    axis.set_title(f"Forecast error vs representation payload\nrun={run_id}", fontsize=10.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__error-payload-tradeoff.png", dpi=dpi)
    plt.close(figure)


def _display_representation(name: str) -> str:
    return {"raw": "Raw", "first_difference": "First difference", "vectorchain": "VectorChain"}[
        name
    ]


def _short_signal(name: str) -> str:
    return {
        "sine": "Sine",
        "chirp": "Chirp",
        "ramp": "Ramp",
        "piecewise_linear": "Piecewise",
        "first_order_response": "First-order",
        "second_order_response": "Second-order",
        "regime_change": "Regime",
    }[name]


def _validate_design(
    inputs: NDArray[np.float64], *, expected_features: int | None = None
) -> NDArray[np.float64]:
    values = np.asarray(inputs)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        msg = "inputs must be a non-empty two-dimensional design matrix"
        raise ValueError(msg)
    if expected_features is not None and values.shape[1] != expected_features:
        msg = f"inputs must have exactly {expected_features} features"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        msg = "inputs must contain real numeric values"
        raise TypeError(msg)
    validated = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated)):
        msg = "inputs must contain only finite values"
        raise ValueError(msg)
    return validated


def _models_equal(left: RidgeModel, right: RidgeModel) -> bool:
    return (
        np.array_equal(left.mean_, right.mean_)
        and np.array_equal(left.scale_, right.scale_)
        and np.array_equal(left.coefficients_, right.coefficients_)
        and left.intercept_ == right.intercept_
    )


def _quartiles(durations: Sequence[float]) -> tuple[float, float, float]:
    values = np.asarray(durations, dtype=np.float64)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def _split_boundaries(config: ForecastConfig) -> tuple[int, int]:
    train_end = math.floor(config.n_points * config.train_fraction)
    validation_end = math.floor(
        config.n_points * (config.train_fraction + config.validation_fraction)
    )
    return train_end, validation_end


def _validate_split_counts(config: ForecastConfig) -> None:
    train_end, validation_end = _split_boundaries(config)
    first_target = config.context_length + config.horizon - 1
    targets = tuple(range(first_target, config.n_points, config.stride))
    counts = (
        sum(target < train_end for target in targets),
        sum(train_end <= target < validation_end for target in targets),
        sum(target >= validation_end for target in targets),
    )
    if any(count == 0 for count in counts):
        msg = "configured temporal split must contain train, validation, and test examples"
        raise ValueError(msg)


def _derive_seed(base_seed: int, signal_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{signal_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _git_state(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _environment_manifest(
    *,
    run_id: str,
    started: datetime,
    git_commit: str,
    git_dirty: bool,
    config: Mapping[str, Any],
    signal_seeds: Mapping[str, int],
    command_args: Sequence[str],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "running",
        "started_utc": started.isoformat(),
        "git": {"commit": git_commit, "dirty": git_dirty},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
        },
        "dependencies": {
            "vectorchain": version("vectorchain"),
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
        },
        "config": config,
        "seeds": {"base": config["experiment"]["seed"], "signals": dict(signal_seeds)},
        "command": {"argv": list(command_args), "display": " ".join(command_args)},
    }


def _write_manifest(run_dir: Path, *, status: str) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(candidate for candidate in run_dir.rglob("*") if candidate.is_file()):
        if path.name == "manifest.json":
            continue
        content = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    _write_json(run_dir / "manifest.json", {"status": status, "files": files})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        msg = f"{name} must be a TOML table"
        raise ValueError(msg)
    return value


def _signal_parameter_tables(
    value: object, signal_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict) or set(value) != set(signal_names):
        msg = "signals.parameters must contain exactly one table for every configured signal"
        raise ValueError(msg)
    validated: dict[str, dict[str, float]] = {}
    for signal_name in signal_names:
        parameters = value[signal_name]
        if not isinstance(parameters, dict) or set(parameters) != SIGNAL_PARAMETERS[signal_name]:
            expected = tuple(sorted(SIGNAL_PARAMETERS[signal_name]))
            msg = f"signals.parameters.{signal_name} must contain exactly {expected}"
            raise ValueError(msg)
        validated[signal_name] = {
            key: _real(parameter, f"signals.parameters.{signal_name}.{key}")
            for key, parameter in parameters.items()
        }
    return validated


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _unique_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty list of strings"
        raise ValueError(msg)
    if any(not isinstance(item, str) for item in value):
        msg = f"{name} must contain only strings"
        raise TypeError(msg)
    result = tuple(value)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicates"
        raise ValueError(msg)
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    result = int(value)
    if result < minimum:
        msg = f"{name} must be greater than or equal to {minimum}"
        raise ValueError(msg)
    return result


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a finite real number"
        raise TypeError(msg)
    result = float(value)
    if not np.isfinite(result):
        msg = f"{name} must be a finite real number"
        raise ValueError(msg)
    if minimum is not None and result < minimum:
        msg = f"{name} must be greater than or equal to {minimum}"
        raise ValueError(msg)
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{name} must be a bool"
        raise TypeError(msg)
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/baseline.toml"),
        help="TOML minimal forecasting configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line forecasting benchmark entry point."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("04_forecasting.py", *argv))
    summary = run_experiment(
        arguments.config,
        output_root=arguments.output_root,
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "run_dir": str(summary.run_dir),
                "n_conditions": summary.n_conditions,
                "n_failures": summary.n_failures,
            },
            sort_keys=True,
        )
    )
    return 1 if summary.n_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
