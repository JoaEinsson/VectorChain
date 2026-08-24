"""Train/validation-only selection runner for the pre-registered K7 experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

import revisable_chain

SELECTION_REPRESENTATION = "revisable_absolute"
LAMBDA_GRID = (0.01, 0.1, 1.0)
RIDGE_ALPHA = 0.001
TRAIN_FRACTION = 0.5
VALIDATION_FRACTION = 0.2
INNER_TRAIN_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Validated K7 configuration that cannot authorize the closed test split."""

    name: str
    phase: str
    scope: str
    seeds: tuple[int, ...]
    mechanisms: tuple[str, ...]
    n_points: int
    noise_std: float
    train_fraction: float
    validation_fraction: float
    inner_train_fraction: float
    lambda_revision: tuple[float, ...]
    lambda_bend: tuple[float, ...]
    selection_representation: str
    model_kind: str
    alpha: float
    output_root: str
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MultiRidgeModel:
    """Training-standardized deterministic multioutput ridge model."""

    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]
    coefficients_: NDArray[np.float64]
    intercept_: NDArray[np.float64]

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict all registered horizons without mutating learned state."""

        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.mean_.size:
            msg = "inputs have the wrong shape for this multioutput ridge"
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = "inputs must be finite"
            raise ValueError(msg)
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
        """Return bytes occupied by learned input standardization state."""

        return self.mean_.nbytes + self.scale_.nbytes

    @property
    def state_bytes(self) -> int:
        """Return bytes occupied by scaler, coefficients and intercepts."""

        return self.scaler_state_bytes + self.coefficients_.nbytes + self.intercept_.nbytes


@dataclass(frozen=True, slots=True)
class ValidationRunSummary:
    """Identity and outputs of one non-decisory K7 validation run."""

    run_id: str
    run_dir: Path
    selection_path: Path
    metrics_path: Path
    selected_lambda_revision: float
    selected_lambda_bend: float
    n_failures: int


def load_config(path: Path) -> ValidationConfig:
    """Load a development-only K7 configuration and reject test authority."""

    resolved_path = path.resolve()
    with resolved_path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = _table(raw, "experiment")
    signals = _table(raw, "signals")
    split = _table(raw, "split")
    selection = _table(raw, "selection")
    model = _table(raw, "model")
    output = _table(raw, "output")

    name = _non_empty_string(experiment.get("name"), "experiment.name")
    phase = _non_empty_string(experiment.get("phase"), "experiment.phase")
    scope = _non_empty_string(experiment.get("scope"), "experiment.scope")
    if scope not in {"development", "fixture"}:
        msg = "experiment.scope must be 'development' or 'fixture'; closed test is unsupported"
        raise ValueError(msg)
    seeds = _integer_tuple(experiment.get("seeds"), "experiment.seeds")
    mechanisms = _string_tuple(signals.get("names"), "signals.names")
    if any(mechanism not in revisable_chain.MECHANISM_NAMES for mechanism in mechanisms):
        msg = "signals.names contains an unregistered K7 mechanism"
        raise ValueError(msg)
    n_points = _integer(signals.get("n_points"), "signals.n_points", minimum=512)
    noise_std = _real(signals.get("noise_std"), "signals.noise_std", minimum=0.0)

    train_fraction = _real(split.get("train_fraction"), "split.train_fraction", minimum=0.0)
    validation_fraction = _real(
        split.get("validation_fraction"), "split.validation_fraction", minimum=0.0
    )
    inner_train_fraction = _real(
        split.get("inner_train_fraction"), "split.inner_train_fraction", minimum=0.0
    )
    if (train_fraction, validation_fraction, inner_train_fraction) != (
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        INNER_TRAIN_FRACTION,
    ):
        msg = "split fractions must exactly match the registered K7 protocol"
        raise ValueError(msg)

    lambda_revision = _real_tuple(selection.get("lambda_revision"), "selection.lambda_revision")
    lambda_bend = _real_tuple(selection.get("lambda_bend"), "selection.lambda_bend")
    selection_representation = _non_empty_string(
        selection.get("representation"), "selection.representation"
    )
    if lambda_revision != LAMBDA_GRID or lambda_bend != LAMBDA_GRID:
        msg = "selection regularizer grids must exactly match the registered K7 protocol"
        raise ValueError(msg)
    if selection_representation != SELECTION_REPRESENTATION:
        msg = f"selection.representation must equal {SELECTION_REPRESENTATION!r}"
        raise ValueError(msg)

    model_kind = _non_empty_string(model.get("kind"), "model.kind")
    alpha = _real(model.get("alpha"), "model.alpha", minimum=0.0)
    if model_kind != "ridge" or alpha != RIDGE_ALPHA:
        msg = "model must be the registered ridge with alpha=0.001"
        raise ValueError(msg)
    output_root = _non_empty_string(output.get("root"), "output.root")

    if scope == "development":
        if seeds != revisable_chain.DEVELOPMENT_SEEDS:
            msg = "development scope must use exactly seeds 11 and 22"
            raise ValueError(msg)
        if mechanisms != revisable_chain.MECHANISM_NAMES:
            msg = "development scope must use all three registered mechanisms"
            raise ValueError(msg)
        if n_points != revisable_chain.N_POINTS or noise_std != revisable_chain.NOISE_STD:
            msg = "development scope must use the registered K7 signal length and noise"
            raise ValueError(msg)
    elif any(seed not in revisable_chain.DEVELOPMENT_SEEDS for seed in seeds):
        msg = "fixture scope is restricted to development seeds 11 and 22"
        raise ValueError(msg)

    return ValidationConfig(
        name=name,
        phase=phase,
        scope=scope,
        seeds=seeds,
        mechanisms=mechanisms,
        n_points=n_points,
        noise_std=noise_std,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        inner_train_fraction=inner_train_fraction,
        lambda_revision=lambda_revision,
        lambda_bend=lambda_bend,
        selection_representation=selection_representation,
        model_kind=model_kind,
        alpha=alpha,
        output_root=output_root,
        raw=raw,
    )


def fit_multi_ridge(
    inputs: NDArray[np.float64], targets: NDArray[np.float64], *, alpha: float
) -> MultiRidgeModel:
    """Fit deterministic multioutput ridge using training-only standardization."""

    values = np.asarray(inputs, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        msg = "inputs must be a non-empty two-dimensional matrix"
        raise ValueError(msg)
    if expected.ndim != 2 or expected.shape != (values.shape[0], len(revisable_chain.HORIZONS)):
        msg = "targets must align with inputs and the registered horizons"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(expected)):
        msg = "ridge inputs and targets must be finite"
        raise ValueError(msg)
    if not np.isfinite(alpha) or alpha <= 0.0:
        msg = "alpha must be finite and positive"
        raise ValueError(msg)

    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (values - mean) / scale
    intercept = np.mean(expected, axis=0)
    coefficients = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(values.shape[1]),
        standardized.T @ (expected - intercept),
    )
    arrays = tuple(
        np.asarray(array, dtype=np.float64) for array in (mean, scale, coefficients, intercept)
    )
    for array in arrays:
        array.flags.writeable = False
    return MultiRidgeModel(arrays[0], arrays[1], arrays[2], arrays[3])


def endpoint_mask(
    target_indices: NDArray[np.int64], *, start_inclusive: int, end_exclusive: int
) -> NDArray[np.bool_]:
    """Select rows only when every horizon endpoint lies inside one split."""

    indices = np.asarray(target_indices)
    if indices.ndim != 2 or indices.shape[1] != len(revisable_chain.HORIZONS):
        msg = "target_indices must contain one column per registered horizon"
        raise ValueError(msg)
    if start_inclusive < 0 or end_exclusive <= start_inclusive:
        msg = "split bounds must define a non-empty non-negative interval"
        raise ValueError(msg)
    return np.asarray(
        np.all(indices >= start_inclusive, axis=1) & np.all(indices < end_exclusive, axis=1),
        dtype=np.bool_,
    )


def run_validation(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] = (),
) -> ValidationRunSummary:
    """Select regularizers and evaluate validation without generating the test suffix."""

    config = load_config(config_path)
    repository_root = Path(__file__).resolve().parents[1]
    started = datetime.now(UTC)
    config_hash = hashlib.sha256(Path(config_path).resolve().read_bytes()).hexdigest()
    run_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash[:10]}"
    root = output_root if output_root is not None else repository_root / config.output_root
    run_dir = Path(root).resolve() / config.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    train_end, validation_end, inner_end = _split_bounds(config)
    git_commit, git_dirty = _git_state(repository_root)
    environment = _environment(
        config=config,
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        command_args=command_args,
        train_end=train_end,
        validation_end=validation_end,
        inner_end=inner_end,
    )
    _write_json(
        run_dir / "config.json", _resolved_config(config, train_end, validation_end, inner_end)
    )
    _write_json(run_dir / "environment.json", environment)
    _write_csv(
        run_dir / "failures.csv", ("stage", "mechanism", "seed", "error_type", "message"), ()
    )
    run_started = time.perf_counter_ns()

    try:
        selected, selection_rows, selection_summary = _select_regularizers(
            config, train_end=train_end, inner_end=inner_end
        )
        selection_payload = {
            "status": f"{config.scope}_train_only",
            "decisory": False,
            "test_materialized": False,
            "selection_representation": config.selection_representation,
            "aggregation": "arithmetic mean of per-series per-horizon NRMSE",
            "nrmse_scale": "inner-fit target population standard deviation, floor 1e-12",
            "tie_break": "larger lambda_revision, then larger lambda_bend within 1e-12",
            "selected": {
                "lambda_revision": selected[0],
                "lambda_bend": selected[1],
                "global_nrmse": selection_summary[selected],
            },
            "grid": [
                {
                    "lambda_revision": pair[0],
                    "lambda_bend": pair[1],
                    "global_nrmse": score,
                }
                for pair, score in selection_summary.items()
            ],
        }
        _write_json(run_dir / "selection.json", selection_payload)
        _write_csv(
            run_dir / "selection_metrics.csv",
            (
                "lambda_revision",
                "lambda_bend",
                "mechanism",
                "seed",
                "horizon",
                "n_inner_fit",
                "n_inner_selection",
                "rmse",
                "training_target_std",
                "nrmse",
            ),
            selection_rows,
        )
        models, artifact_rows = _fit_and_validate(
            config,
            selected=selected,
            train_end=train_end,
            validation_end=validation_end,
        )
        _persist_validation_artifacts(run_dir, models=models, rows=artifact_rows)
    except Exception as error:
        _write_csv(
            run_dir / "failures.csv",
            ("stage", "mechanism", "seed", "error_type", "message"),
            (
                {
                    "stage": "train_validation_runner",
                    "mechanism": "",
                    "seed": "",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            ),
        )
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_started) / 1e9
        environment["n_failures"] = 1
        _write_json(run_dir / "environment.json", environment)
        _write_manifest(run_dir, status="failed")
        raise

    environment["status"] = "complete"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_started) / 1e9
    environment["n_series"] = len(config.seeds) * len(config.mechanisms)
    environment["n_failures"] = 0
    environment["test_materialized"] = False
    environment["selected_lambda_revision"] = selected[0]
    environment["selected_lambda_bend"] = selected[1]
    _write_json(run_dir / "environment.json", environment)
    _write_manifest(run_dir, status="complete")
    return ValidationRunSummary(
        run_id=run_id,
        run_dir=run_dir,
        selection_path=run_dir / "selection.json",
        metrics_path=run_dir / "metrics.csv",
        selected_lambda_revision=selected[0],
        selected_lambda_bend=selected[1],
        n_failures=0,
    )


def _select_regularizers(
    config: ValidationConfig, *, train_end: int, inner_end: int
) -> tuple[
    tuple[float, float],
    list[dict[str, object]],
    dict[tuple[float, float], float],
]:
    rows: list[dict[str, object]] = []
    summary: dict[tuple[float, float], float] = {}
    for lambda_revision in config.lambda_revision:
        for lambda_bend in config.lambda_bend:
            pair_scores: list[float] = []
            for mechanism in config.mechanisms:
                for seed in config.seeds:
                    signal = revisable_chain.generate_k7_signal(
                        mechanism,
                        seed=seed,
                        n_points=config.n_points,
                        noise_std=config.noise_std,
                        stop_exclusive=train_end,
                    )
                    bundle = revisable_chain.build_k7_designs(
                        signal,
                        lambda_revision=lambda_revision,
                        lambda_bend=lambda_bend,
                    )
                    fit_mask = endpoint_mask(
                        bundle.target_indices, start_inclusive=0, end_exclusive=inner_end
                    )
                    selection_mask = endpoint_mask(
                        bundle.target_indices,
                        start_inclusive=inner_end,
                        end_exclusive=train_end,
                    )
                    _require_examples(fit_mask, selection_mask, mechanism, seed)
                    design = bundle.representation(config.selection_representation).inputs
                    model = fit_multi_ridge(
                        design[fit_mask], bundle.targets[fit_mask], alpha=config.alpha
                    )
                    predictions = model.predict(design[selection_mask])
                    for column, horizon in enumerate(revisable_chain.HORIZONS):
                        residual = predictions[:, column] - bundle.targets[selection_mask, column]
                        rmse = float(np.sqrt(np.mean(residual**2)))
                        scale = max(float(np.std(bundle.targets[fit_mask, column])), 1e-12)
                        nrmse = rmse / scale
                        pair_scores.append(nrmse)
                        rows.append(
                            {
                                "lambda_revision": lambda_revision,
                                "lambda_bend": lambda_bend,
                                "mechanism": mechanism,
                                "seed": seed,
                                "horizon": horizon,
                                "n_inner_fit": int(np.count_nonzero(fit_mask)),
                                "n_inner_selection": int(np.count_nonzero(selection_mask)),
                                "rmse": rmse,
                                "training_target_std": scale,
                                "nrmse": nrmse,
                            }
                        )
            summary[(lambda_revision, lambda_bend)] = float(np.mean(pair_scores))

    best_score = min(summary.values())
    tied = [pair for pair, score in summary.items() if abs(score - best_score) <= 1e-12]
    selected = max(tied, key=lambda pair: (pair[0], pair[1]))
    return selected, rows, summary


def _fit_and_validate(
    config: ValidationConfig,
    *,
    selected: tuple[float, float],
    train_end: int,
    validation_end: int,
) -> tuple[dict[str, MultiRidgeModel], dict[str, list[dict[str, object]]]]:
    models: dict[str, MultiRidgeModel] = {}
    rows: dict[str, list[dict[str, object]]] = {
        "origins": [],
        "inputs": [],
        "predictions": [],
        "metrics": [],
        "working_state": [],
        "commit_audit": [],
        "causality_audit": [],
        "solver_audit": [],
    }
    for mechanism in config.mechanisms:
        for seed in config.seeds:
            signal = revisable_chain.generate_k7_signal(
                mechanism,
                seed=seed,
                n_points=config.n_points,
                noise_std=config.noise_std,
                stop_exclusive=validation_end,
            )
            bundle = revisable_chain.build_k7_designs(
                signal,
                lambda_revision=selected[0],
                lambda_bend=selected[1],
            )
            train_mask = endpoint_mask(
                bundle.target_indices, start_inclusive=0, end_exclusive=train_end
            )
            validation_mask = endpoint_mask(
                bundle.target_indices,
                start_inclusive=train_end,
                end_exclusive=validation_end,
            )
            _require_examples(train_mask, validation_mask, mechanism, seed)
            _append_origin_rows(
                rows["origins"], bundle, mechanism, seed, train_mask, validation_mask
            )
            _append_structural_rows(rows, bundle, mechanism, seed, train_end)

            for design in bundle.representations:
                _append_input_rows(
                    rows["inputs"], design, bundle, mechanism, seed, train_mask, validation_mask
                )
                if design.name == "persistence":
                    predictions = np.zeros(
                        (int(np.count_nonzero(validation_mask)), len(revisable_chain.HORIZONS)),
                        dtype=np.float64,
                    )
                    input_rank = 0
                    scaler_state_bytes = 0
                    model_state_bytes = 0
                    train_runtime_s = 0.0
                else:
                    train_started = time.perf_counter_ns()
                    model = fit_multi_ridge(
                        design.inputs[train_mask], bundle.targets[train_mask], alpha=config.alpha
                    )
                    train_runtime_s = (time.perf_counter_ns() - train_started) / 1e9
                    if model.n_predictive_parameters != revisable_chain.N_PREDICTIVE_PARAMETERS:
                        msg = "trained representation violated registered parameter parity"
                        raise RuntimeError(msg)
                    key = f"{mechanism}__seed_{seed}__{design.name}"
                    models[key] = model
                    predictions = model.predict(design.inputs[validation_mask])
                    input_rank = int(np.linalg.matrix_rank(design.inputs[train_mask]))
                    scaler_state_bytes = model.scaler_state_bytes
                    model_state_bytes = model.state_bytes
                _append_prediction_metric_rows(
                    rows["predictions"],
                    rows["metrics"],
                    bundle,
                    design.name,
                    mechanism,
                    seed,
                    validation_mask,
                    predictions,
                    input_rank=input_rank,
                    scaler_state_bytes=scaler_state_bytes,
                    model_state_bytes=model_state_bytes,
                    train_runtime_s=train_runtime_s,
                )
            _append_causality_row(
                rows["causality_audit"],
                bundle,
                mechanism,
                seed,
                selected,
            )
    return models, rows


def _append_origin_rows(
    rows: list[dict[str, object]],
    bundle: revisable_chain.K7DesignBundle,
    mechanism: str,
    seed: int,
    train_mask: NDArray[np.bool_],
    validation_mask: NDArray[np.bool_],
) -> None:
    for row_index, origin in enumerate(bundle.origins):
        split = _row_split(row_index, train_mask, validation_mask)
        rows.append(
            {
                "mechanism": mechanism,
                "seed": seed,
                "row_index": row_index,
                "origin": int(origin),
                "split": split,
                "target_1": int(bundle.target_indices[row_index, 0]),
                "target_8": int(bundle.target_indices[row_index, 1]),
                "target_32": int(bundle.target_indices[row_index, 2]),
                "link_ids": "|".join(str(value) for value in bundle.link_ids[row_index]),
                "boundaries": "|".join(
                    f"{start}:{end}" for start, end in bundle.boundaries[row_index]
                ),
            }
        )


def _append_input_rows(
    rows: list[dict[str, object]],
    design: revisable_chain.RepresentationDesign,
    bundle: revisable_chain.K7DesignBundle,
    mechanism: str,
    seed: int,
    train_mask: NDArray[np.bool_],
    validation_mask: NDArray[np.bool_],
) -> None:
    for row_index, origin in enumerate(bundle.origins):
        values = design.inputs[row_index] if design.inputs.shape[1] else np.asarray(())
        row: dict[str, object] = {
            "mechanism": mechanism,
            "seed": seed,
            "representation": design.name,
            "row_index": row_index,
            "origin": int(origin),
            "split": _row_split(row_index, train_mask, validation_mask),
        }
        row.update(
            {
                f"feature_{index:02d}": float(values[index]) if index < values.size else ""
                for index in range(revisable_chain.N_INPUT_SCALARS)
            }
        )
        rows.append(row)


def _append_prediction_metric_rows(
    prediction_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    bundle: revisable_chain.K7DesignBundle,
    representation: str,
    mechanism: str,
    seed: int,
    validation_mask: NDArray[np.bool_],
    predictions: NDArray[np.float64],
    *,
    input_rank: int,
    scaler_state_bytes: int,
    model_state_bytes: int,
    train_runtime_s: float,
) -> None:
    selected_rows = np.flatnonzero(validation_mask)
    actual = bundle.targets[validation_mask]
    for local_index, row_index in enumerate(selected_rows):
        for column, horizon in enumerate(revisable_chain.HORIZONS):
            prediction_rows.append(
                {
                    "mechanism": mechanism,
                    "seed": seed,
                    "representation": representation,
                    "split": "validation",
                    "origin": int(bundle.origins[row_index]),
                    "horizon": horizon,
                    "target_index": int(bundle.target_indices[row_index, column]),
                    "actual": float(actual[local_index, column]),
                    "predicted": float(predictions[local_index, column]),
                }
            )
    for column, horizon in enumerate(revisable_chain.HORIZONS):
        residual = predictions[:, column] - actual[:, column]
        metric_rows.append(
            {
                "mechanism": mechanism,
                "seed": seed,
                "representation": representation,
                "split": "validation",
                "horizon": horizon,
                "n_origins": predictions.shape[0],
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "input_steps": 0
                if representation == "persistence"
                else (16 if representation == "raw_matched" else 4),
                "input_scalars": 0
                if representation == "persistence"
                else revisable_chain.N_INPUT_SCALARS,
                "input_bytes": 0
                if representation == "persistence"
                else revisable_chain.N_INPUT_SCALARS * np.dtype(np.float64).itemsize,
                "input_rank": input_rank,
                "n_predictive_parameters": 0
                if representation == "persistence"
                else revisable_chain.N_PREDICTIVE_PARAMETERS,
                "scaler_state_bytes": scaler_state_bytes,
                "model_state_bytes": model_state_bytes,
                "train_runtime_s": train_runtime_s,
            }
        )


def _append_structural_rows(
    rows: dict[str, list[dict[str, object]]],
    bundle: revisable_chain.K7DesignBundle,
    mechanism: str,
    seed: int,
    train_end: int,
) -> None:
    training_radii = [link.r for item in bundle.versions[:train_end] for link in item.links]
    median_r = max(float(np.median(training_radii)), 1e-12)
    derivative = np.abs(bundle.signal.latent_derivative[:train_end])
    lower, upper = np.quantile(derivative, (0.25, 0.75))
    values = bundle.signal.values
    committed_anchors = {link.end_joint_id: link.end_value for link in bundle.committed}
    for item in bundle.versions:
        revised_rmse, immutable_rmse = _tail_reconstruction_rmse(values, item)
        first_joint = item.joints[0]
        expected_start = committed_anchors.get(
            first_joint.joint_id, float(values[first_joint.sample_index])
        )
        endpoint_start_error = abs(first_joint.value - expected_start)
        endpoint_current_error = abs(item.joints[-1].value - values[item.observed_at])
        rows["solver_audit"].append(
            {
                "mechanism": mechanism,
                "seed": seed,
                "observed_at": item.observed_at,
                "condition_number": item.condition_number,
                "n_links": len(item.links),
                "raw_span": item.raw_span,
                "immutable_tail_rmse": immutable_rmse,
                "revised_tail_rmse": revised_rmse,
                "start_anchor_error": endpoint_start_error,
                "current_anchor_error": endpoint_current_error,
                "structural_pass": int(
                    len(item.links) <= 4
                    and item.raw_span <= 256
                    and endpoint_start_error <= 1e-12
                    and endpoint_current_error <= 1e-12
                    and np.isfinite(item.condition_number)
                ),
            }
        )
        latent_speed = float(abs(bundle.signal.latent_derivative[item.observed_at]))
        region = (
            "changing"
            if latent_speed >= upper
            else "stationary"
            if latent_speed <= lower
            else "middle"
        )
        for link in item.links:
            rows["working_state"].append(
                {
                    "mechanism": mechanism,
                    "seed": seed,
                    "observed_at": item.observed_at,
                    "version": item.version,
                    "link_id": link.link_id,
                    "created_at": link.created_at,
                    "start": link.start,
                    "end": link.end,
                    "start_value": link.start_value,
                    "end_value": link.end_value,
                    "dt": link.dt,
                    "dy": link.dy,
                    "theta": link.theta,
                    "r": link.r,
                    "update_theta": link.update_theta,
                    "update_r": link.update_r,
                    "correction_energy": float(
                        np.hypot(link.update_theta, link.update_r / median_r)
                    ),
                    "latent_speed": latent_speed,
                    "latent_region": region,
                }
            )
    for link in bundle.committed:
        rows["commit_audit"].append(
            {
                "mechanism": mechanism,
                "seed": seed,
                "link_id": link.link_id,
                "created_at": link.created_at,
                "committed_at": link.committed_at,
                "start": link.start,
                "end": link.end,
                "start_value": link.start_value,
                "end_value": link.end_value,
                "immutable_snapshot": 1,
            }
        )


def _append_causality_row(
    rows: list[dict[str, object]],
    original: revisable_chain.K7DesignBundle,
    mechanism: str,
    seed: int,
    selected: tuple[float, float],
) -> None:
    cut = max(
        revisable_chain.RAW_MATCHED_STEPS + max(revisable_chain.HORIZONS) + 1,
        len(original.signal.values) // 2,
    )
    changed = np.array(original.signal.values, copy=True)
    suffix = np.arange(changed.size - cut - 1, dtype=np.float64)
    changed[cut + 1 :] = changed[cut + 1 :] + 7.0 + np.sin(suffix)
    changed.flags.writeable = False
    modified_signal = replace(original.signal, values=changed)
    modified = revisable_chain.build_k7_designs(
        modified_signal,
        lambda_revision=selected[0],
        lambda_bend=selected[1],
    )
    version_equal = original.versions[cut] == modified.versions[cut]
    original_committed = tuple(link for link in original.committed if link.committed_at <= cut)
    modified_committed = tuple(link for link in modified.committed if link.committed_at <= cut)
    committed_equal = original_committed == modified_committed
    if not version_equal or not committed_equal:
        msg = f"causality suffix audit failed for {mechanism}/seed={seed}"
        raise RuntimeError(msg)
    rows.append(
        {
            "mechanism": mechanism,
            "seed": seed,
            "cut": cut,
            "modified_suffix_start": cut + 1,
            "same_working_version_at_cut": int(version_equal),
            "same_committed_prefix_at_cut": int(committed_equal),
            "passed": 1,
        }
    )


def _tail_reconstruction_rmse(
    values: NDArray[np.float64], item: revisable_chain.WorkingVersion
) -> tuple[float, float]:
    times = np.asarray([joint.sample_index for joint in item.joints], dtype=np.float64)
    revised = np.asarray([joint.value for joint in item.joints], dtype=np.float64)
    immutable = values[times.astype(np.int64)]
    samples = np.arange(int(times[0]), int(times[-1]) + 1)
    raw = values[samples]
    revised_error = np.interp(samples, times, revised) - raw
    immutable_error = np.interp(samples, times, immutable) - raw
    return (
        float(np.sqrt(np.mean(revised_error**2))),
        float(np.sqrt(np.mean(immutable_error**2))),
    )


def _persist_validation_artifacts(
    run_dir: Path,
    *,
    models: Mapping[str, MultiRidgeModel],
    rows: Mapping[str, list[dict[str, object]]],
) -> None:
    _write_csv_from_rows(run_dir / "origins.csv", rows["origins"])
    _write_csv_from_rows(run_dir / "predictions.csv", rows["predictions"])
    _write_csv_from_rows(run_dir / "metrics.csv", rows["metrics"])
    _write_csv_from_rows(run_dir / "solver_audit.csv", rows["solver_audit"])
    _write_csv_from_rows(run_dir / "commit_audit.csv", rows["commit_audit"])
    _write_csv_from_rows(run_dir / "causality_audit.csv", rows["causality_audit"])
    _write_gzip_csv_from_rows(run_dir / "inputs.csv.gz", rows["inputs"])
    _write_gzip_csv_from_rows(run_dir / "working_state.csv.gz", rows["working_state"])
    arrays: dict[str, NDArray[np.float64]] = {}
    for key, model in models.items():
        arrays[f"{key}__mean"] = model.mean_
        arrays[f"{key}__scale"] = model.scale_
        arrays[f"{key}__coefficients"] = model.coefficients_
        arrays[f"{key}__intercept"] = model.intercept_
    np.savez_compressed(run_dir / "models.npz", **arrays)


def _row_split(
    row_index: int, train_mask: NDArray[np.bool_], validation_mask: NDArray[np.bool_]
) -> str:
    if train_mask[row_index]:
        return "train"
    if validation_mask[row_index]:
        return "validation"
    return "boundary_excluded"


def _require_examples(
    first: NDArray[np.bool_], second: NDArray[np.bool_], mechanism: str, seed: int
) -> None:
    if not np.any(first) or not np.any(second):
        msg = f"insufficient endpoint-contained examples for {mechanism}/seed={seed}"
        raise ValueError(msg)


def _split_bounds(config: ValidationConfig) -> tuple[int, int, int]:
    train_end = int(config.n_points * config.train_fraction)
    validation_end = int(config.n_points * (config.train_fraction + config.validation_fraction))
    inner_end = int(train_end * config.inner_train_fraction)
    if validation_end >= config.n_points:
        msg = "validation prefix must leave the closed test suffix unobserved"
        raise ValueError(msg)
    return train_end, validation_end, inner_end


def _resolved_config(
    config: ValidationConfig, train_end: int, validation_end: int, inner_end: int
) -> dict[str, object]:
    return {
        "source": config.raw,
        "resolved": {
            "scope": config.scope,
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
            "test_start_inclusive": validation_end,
            "inner_fit_end_exclusive": inner_end,
            "generated_stop_exclusive": validation_end,
            "test_materialization_supported": False,
            "row_split_rule": "all target endpoints must lie within the same split",
            "model_unit": "one model per mechanism, seed and representation",
        },
    }


def _environment(
    *,
    config: ValidationConfig,
    run_id: str,
    started: datetime,
    git_commit: str,
    git_dirty: bool,
    command_args: Sequence[str],
    train_end: int,
    validation_end: int,
    inner_end: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "running",
        "started_utc": started.isoformat(),
        "phase": config.phase,
        "scope": config.scope,
        "decisory": False,
        "test_materialized": False,
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
        },
        "dependencies": {
            "vectorchain": _package_version("vectorchain"),
            "numpy": np.__version__,
        },
        "config": _resolved_config(config, train_end, validation_end, inner_end),
        "command": {"argv": list(command_args), "display": " ".join(command_args)},
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


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


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        _write_csv_stream(stream, fieldnames, rows)


def _write_csv_from_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        msg = f"cannot infer columns for empty artifact {path.name}"
        raise ValueError(msg)
    _write_csv(path, tuple(rows[0]), rows)


def _write_gzip_csv_from_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        msg = f"cannot infer columns for empty artifact {path.name}"
        raise ValueError(msg)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        _write_csv_stream(stream, tuple(rows[0]), rows)


def _write_csv_stream(
    stream: TextIO,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        msg = f"{name} must be a TOML table"
        raise ValueError(msg)
    return value


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    validated = int(value)
    if validated < minimum:
        msg = f"{name} must be greater than or equal to {minimum}"
        raise ValueError(msg)
    return validated


def _real(value: object, name: str, *, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a real number"
        raise TypeError(msg)
    validated = float(value)
    if not np.isfinite(validated) or validated < minimum:
        msg = f"{name} must be finite and greater than or equal to {minimum}"
        raise ValueError(msg)
    return validated


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty integer array"
        raise ValueError(msg)
    result = tuple(_integer(item, name, minimum=0) for item in value)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicates"
        raise ValueError(msg)
    return result


def _real_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty real array"
        raise ValueError(msg)
    result = tuple(_real(item, name, minimum=0.0) for item in value)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicates"
        raise ValueError(msg)
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty string array"
        raise ValueError(msg)
    result = tuple(_non_empty_string(item, name) for item in value)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicates"
        raise ValueError(msg)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run K7 regularizer selection and external validation without opening test"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the train/validation-only K7 command."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("09_revisable_chain_validation.py", *argv))
    result = run_validation(
        arguments.config,
        output_root=arguments.output_root,
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "selected_lambda_revision": result.selected_lambda_revision,
                "selected_lambda_bend": result.selected_lambda_bend,
                "test_materialized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
