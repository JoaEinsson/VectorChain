"""Paired controls for the exploratory absolute-geometry forecasting result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vectorchain import fixed_linear_segmentation
from vectorchain.features import (
    FeatureName,
    compute_features_from_displacements,
    validate_feature_names,
)

if __package__:
    from experiments import forecasting, forecasting_robustness
else:
    import forecasting
    import forecasting_robustness

SUPPORTED_CONTROLS = (
    "local_geometry",
    "moving_average_geometry",
    "ewma_geometry",
    "fixed_geometry",
    "abba_geometry",
)
TUNED_CONTROLS = (
    "moving_average_geometry",
    "ewma_geometry",
    "fixed_geometry",
    "abba_geometry",
)

CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    "role",
    "causal_scope",
    "selected_parameter",
    *forecasting.METRIC_FIELDS,
    "candidate_rmse_ratio_vs_control",
    "candidate_predictive_advantage",
    "candidate_pareto_vs_control",
)

SIGNAL_CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    "role",
    "causal_scope",
    "selected_parameter",
    *forecasting.SIGNAL_METRIC_FIELDS,
    "candidate_rmse_ratio_vs_control",
)

TUNING_FIELDS = (
    "seed",
    "horizon",
    "context_length",
    "control",
    "parameter",
    "status",
    "error_type",
    "error_message",
    "n_inner_train",
    "n_inner_validation",
    "inner_validation_rmse",
    "selected",
)

SUMMARY_FIELDS = (
    "representation",
    "role",
    "causal_scope",
    "split",
    "horizon",
    "context_length",
    "tolerance",
    "selected_parameters",
    "n_seeds",
    "n_successes",
    "n_failures",
    "rmse_mean",
    "rmse_median",
    "rmse_ratio_vs_raw_mean",
    "candidate_rmse_ratio_vs_control_mean",
    "candidate_rmse_ratio_vs_control_median",
    "candidate_predictive_advantage_rate",
    "candidate_pareto_rate",
    "robust_predictive_cell",
    "robust_pareto_cell",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "n_pooled_features",
    "n_model_parameters",
    "representation_runtime_median_s",
)

SEED_SUMMARY_FIELDS = (
    "representation",
    "role",
    "causal_scope",
    "split",
    "seed",
    "n_cells",
    "geometric_mean_candidate_rmse_ratio_vs_control",
    "candidate_seed_success",
    "pareto_cell_rate",
)

SIGNAL_SUMMARY_FIELDS = (
    "representation",
    "role",
    "causal_scope",
    "split",
    "horizon",
    "context_length",
    "signal",
    "n_seeds",
    "rmse_mean",
    "rmse_median",
    "candidate_rmse_ratio_vs_control_mean",
    "candidate_rmse_ratio_vs_control_median",
)


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Validated Stage-8 control benchmark configuration."""

    name: str
    base_config_path: Path
    base: forecasting.ForecastConfig
    seeds: tuple[int, ...]
    context_lengths: tuple[int, ...]
    horizons: tuple[int, ...]
    stride: int
    tolerance: float
    candidate_name: str
    candidate_features: tuple[FeatureName, ...]
    control_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    primary_name: str
    inner_train_fraction: float
    moving_average_windows: tuple[int, ...]
    ewma_alphas: tuple[float, ...]
    fixed_segment_lengths: tuple[int, ...]
    abba_tolerances: tuple[float, ...]
    fabba_version: str
    primary_split: str
    maximum_candidate_rmse_ratio: float
    pareto_rmse_ratio: float
    robust_seed_rate: float
    minimum_robust_cell_rate: float
    repetitions: int
    warmup_repetitions: int
    output_root: str
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of one control benchmark run."""

    run_id: str
    run_dir: Path
    summary_path: Path
    gate_path: Path
    n_conditions: int
    n_failures: int
    gate_passed: bool


def load_config(path: Path) -> ControlConfig:
    """Load and validate the registered Stage-8 controls."""

    path = path.resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = forecasting._table(raw, "experiment")
    grid = forecasting._table(raw, "grid")
    candidate_table = forecasting._table(raw, "candidate")
    controls = forecasting._table(raw, "controls")
    parameters = forecasting._table(controls, "parameters")
    external = forecasting._table(raw, "external")
    criteria = forecasting._table(raw, "criteria")
    timing = forecasting._table(raw, "timing")
    output = forecasting._table(raw, "output")

    name = forecasting._non_empty_string(experiment.get("name"), "experiment.name")
    base_reference = forecasting._non_empty_string(
        experiment.get("base_config"), "experiment.base_config"
    )
    base_candidate = Path(base_reference)
    base_path = (
        base_candidate if base_candidate.is_absolute() else path.parent / base_candidate
    ).resolve()
    base = forecasting.load_config(base_path)
    seeds = forecasting_robustness._unique_integer_tuple(
        experiment.get("seeds"), "experiment.seeds", minimum=0
    )
    if len(seeds) < 2:
        raise ValueError("experiment.seeds must contain at least two independent seeds")
    contexts = forecasting_robustness._increasing_integer_tuple(
        grid.get("context_lengths"), "grid.context_lengths", minimum=3
    )
    horizons = forecasting_robustness._increasing_integer_tuple(
        grid.get("horizons"), "grid.horizons", minimum=1
    )
    stride = forecasting._integer(grid.get("stride"), "grid.stride", minimum=1)
    tolerance = forecasting._real(grid.get("tolerance"), "grid.tolerance", minimum=0.0)

    candidate_name = forecasting._non_empty_string(candidate_table.get("name"), "candidate.name")
    if candidate_name != "absolute_geometry":
        raise ValueError("candidate.name must equal 'absolute_geometry'")
    raw_features = candidate_table.get("features")
    if not isinstance(raw_features, list):
        raise TypeError("candidate.features must be a list")
    candidate_features = validate_feature_names(raw_features)
    if candidate_features != ("dt", "dy", "theta", "r"):
        raise ValueError("candidate.features must exactly equal dt, dy, theta, r")

    control_names = forecasting._unique_string_tuple(controls.get("names"), "controls.names")
    if any(item not in SUPPORTED_CONTROLS for item in control_names):
        raise ValueError(f"controls.names must be drawn from {SUPPORTED_CONTROLS}")
    gate_names = forecasting._unique_string_tuple(controls.get("gate_names"), "controls.gate_names")
    if not set(gate_names) <= set(control_names) or "abba_geometry" in gate_names:
        raise ValueError("controls.gate_names must be configured causal controls")
    primary_name = forecasting._non_empty_string(
        controls.get("primary_name"), "controls.primary_name"
    )
    if primary_name not in gate_names:
        raise ValueError("controls.primary_name must belong to controls.gate_names")

    inner_fraction = forecasting._real(
        parameters.get("inner_train_fraction"),
        "controls.parameters.inner_train_fraction",
        minimum=0.0,
    )
    if not 0.5 <= inner_fraction < 1.0:
        raise ValueError("inner_train_fraction must satisfy 0.5 <= value < 1")
    moving_windows = _three_increasing_integers(
        parameters.get("moving_average_windows"),
        "controls.parameters.moving_average_windows",
    )
    ewma_alphas = _three_increasing_reals(
        parameters.get("ewma_alphas"), "controls.parameters.ewma_alphas"
    )
    if ewma_alphas[0] <= 0.0 or ewma_alphas[-1] > 1.0:
        raise ValueError("ewma_alphas must satisfy 0 < alpha <= 1")
    fixed_lengths = _three_increasing_integers(
        parameters.get("fixed_segment_lengths"),
        "controls.parameters.fixed_segment_lengths",
    )
    abba_tolerances = _three_increasing_reals(
        parameters.get("abba_tolerances"), "controls.parameters.abba_tolerances"
    )
    fabba_version = forecasting._non_empty_string(
        external.get("fabba_version"), "external.fabba_version"
    )

    primary_split = forecasting._non_empty_string(
        criteria.get("primary_split"), "criteria.primary_split"
    )
    if primary_split != "validation":
        raise ValueError("criteria.primary_split must equal 'validation'")
    maximum_ratio = forecasting._real(
        criteria.get("maximum_candidate_rmse_ratio"),
        "criteria.maximum_candidate_rmse_ratio",
        minimum=0.0,
    )
    if not 0.0 < maximum_ratio < 1.0:
        raise ValueError("maximum_candidate_rmse_ratio must satisfy 0 < value < 1")
    pareto_ratio = forecasting._real(
        criteria.get("pareto_rmse_ratio"), "criteria.pareto_rmse_ratio", minimum=1.0
    )
    robust_seed_rate = _unit_interval(criteria.get("robust_seed_rate"), "criteria.robust_seed_rate")
    minimum_cell_rate = _unit_interval(
        criteria.get("minimum_robust_cell_rate"), "criteria.minimum_robust_cell_rate"
    )
    repetitions = forecasting._integer(timing.get("repetitions"), "timing.repetitions", minimum=1)
    warmups = forecasting._integer(
        timing.get("warmup_repetitions"), "timing.warmup_repetitions", minimum=0
    )
    output_root = forecasting._non_empty_string(output.get("root"), "output.root")
    save_plots = forecasting._boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = forecasting._integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    if base.context_length not in contexts or base.horizon not in horizons:
        raise ValueError("grid must include the base context and horizon")
    config = ControlConfig(
        name=name,
        base_config_path=base_path,
        base=base,
        seeds=seeds,
        context_lengths=contexts,
        horizons=horizons,
        stride=stride,
        tolerance=tolerance,
        candidate_name=candidate_name,
        candidate_features=candidate_features,
        control_names=control_names,
        gate_names=gate_names,
        primary_name=primary_name,
        inner_train_fraction=inner_fraction,
        moving_average_windows=moving_windows,
        ewma_alphas=ewma_alphas,
        fixed_segment_lengths=fixed_lengths,
        abba_tolerances=abba_tolerances,
        fabba_version=fabba_version,
        primary_split=primary_split,
        maximum_candidate_rmse_ratio=maximum_ratio,
        pareto_rmse_ratio=pareto_ratio,
        robust_seed_rate=robust_seed_rate,
        minimum_robust_cell_rate=minimum_cell_rate,
        repetitions=repetitions,
        warmup_repetitions=warmups,
        output_root=output_root,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )
    for context in config.context_lengths:
        for horizon in config.horizons:
            forecasting._validate_split_counts(
                _condition_config(config, config.seeds[0], context, horizon)
            )
    return config


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute the complete paired-control grid."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    external_versions = _verify_external_dependencies(config)
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

    representations = ("raw", "first_difference", config.candidate_name, *config.control_names)
    expected_conditions = (
        len(config.seeds)
        * len(config.context_lengths)
        * len(config.horizons)
        * len(representations)
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
        "expected_condition_evaluations": expected_conditions,
        "experimental_dependencies": external_versions,
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
    environment["experimental_dependencies"] = external_versions
    forecasting._write_json(run_dir / "environment.json", environment)

    condition_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    derived_seeds: dict[str, dict[str, int]] = {}
    run_start = time.perf_counter_ns()
    try:
        for seed in config.seeds:
            seed_config = replace(
                config.base,
                seed=seed,
                stride=config.stride,
                repetitions=config.repetitions,
                warmup_repetitions=config.warmup_repetitions,
                save_models=False,
                save_plots=False,
            )
            signals, signal_seeds = forecasting._generate_signals(seed_config)
            derived_seeds[str(seed)] = signal_seeds
            for context in config.context_lengths:
                for horizon in config.horizons:
                    cell_config = replace(
                        seed_config,
                        context_length=context,
                        horizon=horizon,
                        vectorchain_tolerance=config.tolerance,
                        vectorchain_features=config.candidate_features,
                    )
                    examples = forecasting._build_examples(signals, signal_seeds, cell_config)
                    cell_rows: list[dict[str, object]] = []
                    cell_signal_rows: list[dict[str, object]] = []

                    for name in ("raw", "first_difference", config.candidate_name):
                        backend = "vectorchain" if name == config.candidate_name else name
                        _evaluate_condition(
                            name=name,
                            backend=backend,
                            selected_parameter="",
                            transform=None,
                            seed=seed,
                            context=context,
                            horizon=horizon,
                            examples=examples,
                            condition_config=cell_config,
                            config=config,
                            condition_rows=cell_rows,
                            signal_rows=cell_signal_rows,
                        )
                        _write_partial(
                            run_dir,
                            (*condition_rows, *cell_rows),
                            (*signal_rows, *cell_signal_rows),
                            tuning_rows,
                        )

                    for control in config.control_names:
                        try:
                            parameter = _selected_parameter(
                                control,
                                examples,
                                cell_config,
                                config,
                                seed=seed,
                                context=context,
                                horizon=horizon,
                                tuning_rows=tuning_rows,
                            )
                            transform = _control_transform(control, parameter, config)
                            _evaluate_condition(
                                name=control,
                                backend=control,
                                selected_parameter=_format_parameter(parameter),
                                transform=transform,
                                seed=seed,
                                context=context,
                                horizon=horizon,
                                examples=examples,
                                condition_config=cell_config,
                                config=config,
                                condition_rows=cell_rows,
                                signal_rows=cell_signal_rows,
                            )
                        except Exception as error:
                            _record_failure(
                                name=control,
                                selected_parameter="",
                                seed=seed,
                                context=context,
                                horizon=horizon,
                                error=error,
                                config=config,
                                condition_rows=cell_rows,
                            )
                        _write_partial(
                            run_dir,
                            (*condition_rows, *cell_rows),
                            (*signal_rows, *cell_signal_rows),
                            tuning_rows,
                        )

                    forecasting._add_raw_comparisons(cell_rows)
                    _add_candidate_comparisons(cell_rows, cell_signal_rows, config)
                    condition_rows.extend(cell_rows)
                    signal_rows.extend(cell_signal_rows)
                    _write_partial(run_dir, condition_rows, signal_rows, tuning_rows)

        summary_rows = _summarize_conditions(condition_rows, config)
        seed_summary_rows = _summarize_seeds(condition_rows, config)
        signal_summary_rows = _summarize_signals(signal_rows)
        gate = _evaluate_gate(condition_rows, seed_summary_rows, config, expected_conditions)
        forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
        forecasting._write_csv(
            run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
        )
        forecasting._write_csv(run_dir / "tuning.csv", TUNING_FIELDS, tuning_rows)
        forecasting._write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        forecasting._write_csv(
            run_dir / "summary_by_seed.csv", SEED_SUMMARY_FIELDS, seed_summary_rows
        )
        forecasting._write_csv(
            run_dir / "summary_by_signal.csv", SIGNAL_SUMMARY_FIELDS, signal_summary_rows
        )
        forecasting._write_json(run_dir / "gate.json", gate)
        if config.save_plots and all(row["status"] == "ok" for row in condition_rows):
            _plot_summaries(
                condition_rows, summary_rows, tuning_rows, gate, config, plots_dir, run_id
            )
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
        environment["derived_signal_seeds"] = derived_seeds
        forecasting._write_json(run_dir / "environment.json", environment)
        forecasting._write_manifest(run_dir, status="failed")
        raise

    failures = {str(row["condition_id"]) for row in condition_rows if row["status"] != "ok"}
    environment["status"] = "complete" if not failures else "complete_with_failures"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
    environment["n_conditions"] = expected_conditions
    environment["n_failures"] = len(failures)
    environment["scientific_gate_passed"] = bool(gate["passed"])
    environment["derived_signal_seeds"] = derived_seeds
    forecasting._write_json(run_dir / "environment.json", environment)
    forecasting._write_manifest(run_dir, status=str(environment["status"]))
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        summary_path=run_dir / "summary.csv",
        gate_path=run_dir / "gate.json",
        n_conditions=expected_conditions,
        n_failures=len(failures),
        gate_passed=bool(gate["passed"]),
    )


def _condition_config(
    config: ControlConfig, seed: int, context: int, horizon: int
) -> forecasting.ForecastConfig:
    return replace(
        config.base,
        seed=seed,
        context_length=context,
        horizon=horizon,
        stride=config.stride,
        vectorchain_tolerance=config.tolerance,
        vectorchain_features=config.candidate_features,
        repetitions=config.repetitions,
        warmup_repetitions=config.warmup_repetitions,
        save_models=False,
        save_plots=False,
    )


def _evaluate_condition(
    *,
    name: str,
    backend: str,
    selected_parameter: str,
    transform: forecasting.SequenceTransform | None,
    seed: int,
    context: int,
    horizon: int,
    examples: Sequence[forecasting.ForecastExample],
    condition_config: forecasting.ForecastConfig,
    config: ControlConfig,
    condition_rows: list[dict[str, object]],
    signal_rows: list[dict[str, object]],
) -> None:
    prefix = _prefix(name, selected_parameter, seed, context, horizon, config)
    try:
        result = forecasting._run_representation(
            backend, examples, condition_config, transform=transform
        )
        condition_rows.extend({**row, **prefix, "representation": name} for row in result[0])
        signal_rows.extend({**row, **prefix, "representation": name} for row in result[1])
    except Exception as error:
        _record_failure(
            name=name,
            selected_parameter=selected_parameter,
            seed=seed,
            context=context,
            horizon=horizon,
            error=error,
            config=config,
            condition_rows=condition_rows,
        )


def _record_failure(
    *,
    name: str,
    selected_parameter: str,
    seed: int,
    context: int,
    horizon: int,
    error: Exception,
    config: ControlConfig,
    condition_rows: list[dict[str, object]],
) -> None:
    prefix = _prefix(name, selected_parameter, seed, context, horizon, config)
    condition_rows.extend(
        {
            **forecasting._failure_row(name, split, error),
            **prefix,
            "representation": name,
        }
        for split in ("validation", "test")
    )


def _prefix(
    name: str,
    parameter: str,
    seed: int,
    context: int,
    horizon: int,
    config: ControlConfig,
) -> dict[str, object]:
    return {
        "condition_id": f"seed-{seed}__h-{horizon}__c-{context}__{name}",
        "seed": seed,
        "horizon": horizon,
        "context_length": context,
        "tolerance": config.tolerance if name == config.candidate_name else "",
        "role": _role(name, config),
        "causal_scope": _causal_scope(name),
        "selected_parameter": parameter,
    }


def _selected_parameter(
    control: str,
    examples: Sequence[forecasting.ForecastExample],
    condition_config: forecasting.ForecastConfig,
    config: ControlConfig,
    *,
    seed: int,
    context: int,
    horizon: int,
    tuning_rows: list[dict[str, object]],
) -> int | float:
    if control == "local_geometry":
        return 1
    candidates = _parameter_candidates(control, config)
    results: list[tuple[float, int | float, dict[str, object]]] = []
    for parameter in candidates:
        row: dict[str, object] = {
            "seed": seed,
            "horizon": horizon,
            "context_length": context,
            "control": control,
            "parameter": _format_parameter(parameter),
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "selected": False,
        }
        try:
            score, n_train, n_validation = _inner_validation_score(
                examples,
                _control_transform(control, parameter, config),
                condition_config,
                config.inner_train_fraction,
            )
            row.update(
                {
                    "n_inner_train": n_train,
                    "n_inner_validation": n_validation,
                    "inner_validation_rmse": score,
                }
            )
            results.append((score, parameter, row))
        except Exception as error:
            row.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "n_inner_train": "",
                    "n_inner_validation": "",
                    "inner_validation_rmse": "",
                }
            )
        tuning_rows.append(row)
    if not results:
        raise RuntimeError(f"all tuning candidates failed for {control}")
    selected = min(results, key=lambda item: (item[0], candidates.index(item[1])))
    selected[2]["selected"] = True
    return selected[1]


def _inner_validation_score(
    examples: Sequence[forecasting.ForecastExample],
    transform: forecasting.SequenceTransform,
    condition_config: forecasting.ForecastConfig,
    inner_train_fraction: float,
) -> tuple[float, int, int]:
    training_examples = [example for example in examples if example.split == "train"]
    summaries = [
        forecasting.summarize_sequence(
            transform(example.context), condition_config.summary_statistics
        )
        for example in training_examples
    ]
    design = np.vstack(summaries)
    targets = np.asarray([example.target_delta for example in training_examples], dtype=np.float64)
    train_mask = np.zeros(len(training_examples), dtype=bool)
    validation_mask = np.zeros(len(training_examples), dtype=bool)
    for signal in condition_config.signal_names:
        indices = [
            index for index, example in enumerate(training_examples) if example.signal == signal
        ]
        if len(indices) < 2:
            raise RuntimeError("inner tuning requires at least two training examples per signal")
        cutoff = max(1, min(len(indices) - 1, math.floor(inner_train_fraction * len(indices))))
        train_mask[indices[:cutoff]] = True
        validation_mask[indices[cutoff:]] = True
    model = forecasting.fit_ridge(
        design[train_mask], targets[train_mask], alpha=condition_config.alpha
    )
    predictions = model.predict(design[validation_mask])
    errors = predictions - targets[validation_mask]
    return (
        float(np.sqrt(np.mean(errors**2))),
        int(np.count_nonzero(train_mask)),
        int(np.count_nonzero(validation_mask)),
    )


def _control_transform(
    control: str, parameter: int | float, config: ControlConfig
) -> forecasting.SequenceTransform:
    if control == "local_geometry":
        return lambda values: _local_geometry(values, config.candidate_features)
    if control == "moving_average_geometry":
        window = int(parameter)
        return lambda values: _local_geometry(
            _trailing_mean(values, window), config.candidate_features
        )
    if control == "ewma_geometry":
        alpha = float(parameter)
        return lambda values: _local_geometry(_ewma(values, alpha), config.candidate_features)
    if control == "fixed_geometry":
        length = int(parameter)
        return lambda values: fixed_linear_segmentation(
            values, segment_length=length, features=config.candidate_features
        ).vectors.copy()
    if control == "abba_geometry":
        tolerance = float(parameter)
        return lambda values: _abba_geometry(values, tolerance, config.candidate_features)
    raise ValueError(f"unsupported control: {control}")


def _local_geometry(
    values: NDArray[np.float64], features: Sequence[FeatureName]
) -> NDArray[np.float64]:
    dy = np.diff(np.asarray(values, dtype=np.float64))
    dt = np.ones(dy.size, dtype=np.float64)
    return compute_features_from_displacements(dt, dy, features)


def _trailing_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(array)))
    result = np.empty_like(array)
    for index in range(array.size):
        start = max(0, index - window + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result


def _ewma(values: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    result[0] = array[0]
    for index in range(1, array.size):
        result[index] = alpha * array[index] + (1.0 - alpha) * result[index - 1]
    return result


def _abba_geometry(
    values: NDArray[np.float64],
    tolerance: float,
    features: Sequence[FeatureName],
) -> NDArray[np.float64]:
    module = import_module("fABBA")
    compress = module.compress
    pieces = np.asarray(
        compress(np.asarray(values, dtype=np.float64), tol=tolerance), dtype=np.float64
    )
    if pieces.ndim != 2 or pieces.shape[0] == 0 or pieces.shape[1] < 2:
        raise RuntimeError("fABBA.compress returned invalid continuous pieces")
    return compute_features_from_displacements(pieces[:, 0], pieces[:, 1], features)


def _add_candidate_comparisons(
    rows: Sequence[dict[str, object]],
    signal_rows: Sequence[dict[str, object]],
    config: ControlConfig,
) -> None:
    candidates = {
        str(row["split"]): row
        for row in rows
        if row["representation"] == config.candidate_name and row["status"] == "ok"
    }
    for row in rows:
        split = str(row["split"])
        if row["status"] != "ok" or row["representation"] not in config.control_names:
            continue
        if split not in candidates:
            continue
        candidate = candidates[split]
        ratio = float(candidate["rmse"]) / float(row["rmse"])
        row["candidate_rmse_ratio_vs_control"] = ratio
        row["candidate_predictive_advantage"] = ratio <= config.maximum_candidate_rmse_ratio
        row["candidate_pareto_vs_control"] = (
            ratio <= config.pareto_rmse_ratio
            and float(candidate["mean_input_scalar_elements"])
            <= float(row["mean_input_scalar_elements"])
            and int(candidate["n_model_parameters"]) <= int(row["n_model_parameters"])
        )
    signal_candidates = {
        (str(row["split"]), str(row["signal"])): row
        for row in signal_rows
        if row["representation"] == config.candidate_name
    }
    for row in signal_rows:
        key = (str(row["split"]), str(row["signal"]))
        if row["representation"] in config.control_names and key in signal_candidates:
            row["candidate_rmse_ratio_vs_control"] = float(signal_candidates[key]["rmse"]) / float(
                row["rmse"]
            )


def _summarize_conditions(
    rows: Sequence[Mapping[str, object]], config: ControlConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["role"],
            row["causal_scope"],
            row["split"],
            row["horizon"],
            row["context_length"],
            row["tolerance"],
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, group in grouped.items():
        successful = [row for row in group if row["status"] == "ok"]
        result: dict[str, object] = {
            "representation": key[0],
            "role": key[1],
            "causal_scope": key[2],
            "split": key[3],
            "horizon": key[4],
            "context_length": key[5],
            "tolerance": key[6],
            "selected_parameters": ",".join(
                sorted(
                    {
                        str(row["selected_parameter"])
                        for row in successful
                        if row["selected_parameter"] != ""
                    }
                )
            ),
            "n_seeds": len({int(row["seed"]) for row in group}),
            "n_successes": len(successful),
            "n_failures": len(group) - len(successful),
        }
        if successful:
            ratios = _optional_values(successful, "candidate_rmse_ratio_vs_control")
            predictive_rate = _boolean_rate(successful, "candidate_predictive_advantage")
            pareto_rate = _boolean_rate(successful, "candidate_pareto_vs_control")
            result.update(
                {
                    "rmse_mean": float(np.mean(_values(successful, "rmse"))),
                    "rmse_median": float(np.median(_values(successful, "rmse"))),
                    "rmse_ratio_vs_raw_mean": _optional_mean(successful, "rmse_ratio_vs_raw"),
                    "candidate_rmse_ratio_vs_control_mean": (
                        float(np.mean(ratios)) if ratios.size else ""
                    ),
                    "candidate_rmse_ratio_vs_control_median": (
                        float(np.median(ratios)) if ratios.size else ""
                    ),
                    "candidate_predictive_advantage_rate": predictive_rate,
                    "candidate_pareto_rate": pareto_rate,
                    "robust_predictive_cell": (
                        predictive_rate != "" and float(predictive_rate) >= config.robust_seed_rate
                    ),
                    "robust_pareto_cell": (
                        pareto_rate != "" and float(pareto_rate) >= config.robust_seed_rate
                    ),
                    "mean_input_steps": float(np.mean(_values(successful, "mean_input_steps"))),
                    "mean_input_scalar_elements": float(
                        np.mean(_values(successful, "mean_input_scalar_elements"))
                    ),
                    "n_pooled_features": int(successful[0]["n_pooled_features"]),
                    "n_model_parameters": int(successful[0]["n_model_parameters"]),
                    "representation_runtime_median_s": float(
                        np.median(_values(successful, "representation_runtime_s"))
                    ),
                }
            )
        output.append({field: result.get(field, "") for field in SUMMARY_FIELDS})
    return sorted(output, key=_summary_sort_key)


def _summarize_seeds(
    rows: Sequence[Mapping[str, object]], config: ControlConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row["representation"] not in config.control_names
            or row["status"] != "ok"
            or row.get("candidate_rmse_ratio_vs_control", "") == ""
        ):
            continue
        key = (
            str(row["representation"]),
            str(row["role"]),
            str(row["causal_scope"]),
            str(row["split"]),
            int(row["seed"]),
        )
        grouped[key].append(row)
    expected = len(config.context_lengths) * len(config.horizons)
    output: list[dict[str, object]] = []
    for key, group in grouped.items():
        geometric = _geometric_mean(_values(group, "candidate_rmse_ratio_vs_control"))
        complete = len(group) == expected
        output.append(
            {
                "representation": key[0],
                "role": key[1],
                "causal_scope": key[2],
                "split": key[3],
                "seed": key[4],
                "n_cells": len(group),
                "geometric_mean_candidate_rmse_ratio_vs_control": geometric,
                "candidate_seed_success": (
                    complete and geometric <= config.maximum_candidate_rmse_ratio
                ),
                "pareto_cell_rate": _boolean_rate(group, "candidate_pareto_vs_control"),
            }
        )
    return sorted(
        output, key=lambda row: (str(row["representation"]), str(row["split"]), int(row["seed"]))
    )


def _summarize_signals(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["role"],
            row["causal_scope"],
            row["split"],
            row["horizon"],
            row["context_length"],
            row["signal"],
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, group in grouped.items():
        ratios = _optional_values(group, "candidate_rmse_ratio_vs_control")
        output.append(
            {
                "representation": key[0],
                "role": key[1],
                "causal_scope": key[2],
                "split": key[3],
                "horizon": key[4],
                "context_length": key[5],
                "signal": key[6],
                "n_seeds": len({int(row["seed"]) for row in group}),
                "rmse_mean": float(np.mean(_values(group, "rmse"))),
                "rmse_median": float(np.median(_values(group, "rmse"))),
                "candidate_rmse_ratio_vs_control_mean": (
                    float(np.mean(ratios)) if ratios.size else ""
                ),
                "candidate_rmse_ratio_vs_control_median": (
                    float(np.median(ratios)) if ratios.size else ""
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["representation"]),
            str(row["split"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["signal"]),
        ),
    )


def _evaluate_gate(
    rows: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    config: ControlConfig,
    expected_conditions: int,
) -> dict[str, object]:
    required_seeds = math.ceil(config.robust_seed_rate * len(config.seeds))
    total_cells = len(config.context_lengths) * len(config.horizons)
    required_cells = math.ceil(config.minimum_robust_cell_rate * total_cells)
    failures = {str(row["condition_id"]) for row in rows if row["status"] != "ok"}
    unique_conditions = {str(row["condition_id"]) for row in rows}
    matched_capacity = all(
        int(row["n_pooled_features"]) == 12 and int(row["n_model_parameters"]) == 13
        for row in rows
        if row["status"] == "ok"
        and row["representation"] in {config.candidate_name, *config.gate_names}
    )
    controls: dict[str, object] = {}
    all_control_checks: list[bool] = []
    for control in config.gate_names:
        selected_seeds = [
            row
            for row in seed_rows
            if row["representation"] == control and row["split"] == config.primary_split
        ]
        seed_successes = sum(row["candidate_seed_success"] is True for row in selected_seeds)
        validation_rows = [
            row
            for row in rows
            if row["representation"] == control
            and row["split"] == config.primary_split
            and row["status"] == "ok"
        ]
        by_cell: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
        for row in validation_rows:
            by_cell[(int(row["context_length"]), int(row["horizon"]))].append(row)
        robust_predictive = sum(
            sum(item["candidate_predictive_advantage"] is True for item in group) >= required_seeds
            for group in by_cell.values()
        )
        robust_pareto = sum(
            sum(item["candidate_pareto_vs_control"] is True for item in group) >= required_seeds
            for group in by_cell.values()
        )
        checks = {
            "complete_seed_grid": len(selected_seeds) == len(config.seeds)
            and all(int(row["n_cells"]) == total_cells for row in selected_seeds),
            "seed_superiority": seed_successes >= required_seeds,
            "robust_predictive_cells": robust_predictive >= required_cells,
            "robust_pareto_cells": robust_pareto >= required_cells,
        }
        all_control_checks.extend(checks.values())
        controls[control] = {
            "observed": {
                "seed_successes": seed_successes,
                "seed_trials": len(selected_seeds),
                "robust_predictive_cells": robust_predictive,
                "robust_pareto_cells": robust_pareto,
                "cell_trials": len(by_cell),
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    execution_checks = {
        "complete_condition_grid": len(unique_conditions) == expected_conditions,
        "zero_condition_failures": not failures,
        "matched_downstream_capacity": matched_capacity,
    }
    return {
        "status": "evaluated",
        "passed": all(execution_checks.values()) and all(all_control_checks),
        "candidate": config.candidate_name,
        "primary_control": config.primary_name,
        "primary_split": config.primary_split,
        "criteria": {
            "maximum_candidate_rmse_ratio": config.maximum_candidate_rmse_ratio,
            "pareto_rmse_ratio": config.pareto_rmse_ratio,
            "robust_seed_rate": config.robust_seed_rate,
            "minimum_robust_cell_rate": config.minimum_robust_cell_rate,
            "required_seeds": required_seeds,
            "required_cells": required_cells,
        },
        "execution_checks": execution_checks,
        "controls": controls,
        "external_descriptive_control": "abba_geometry"
        if "abba_geometry" in config.control_names
        else "",
    }


def _plot_summaries(
    condition_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    tuning_rows: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    config: ControlConfig,
    plots_dir: Path,
    run_id: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharey=True)
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        distributions = []
        for control in config.control_names:
            selected = [
                float(row["candidate_rmse_ratio_vs_control"])
                for row in condition_rows
                if row["representation"] == control and row["split"] == split
            ]
            distributions.append(selected)
        axis.boxplot(
            distributions,
            tick_labels=[
                name.replace("_geometry", "").replace("_", "\n") for name in config.control_names
            ],
            showmeans=True,
        )
        axis.axhline(config.maximum_candidate_rmse_ratio, color="black", linestyle="--")
        axis.axhline(1.0, color="black", linestyle=":")
        axis.set_title(split.title())
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("RMSE(candidate) / RMSE(control)")
    figure.suptitle(f"Paired control distributions\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__control-ratio-distributions.png", dpi=config.plot_dpi)
    plt.close(figure)

    primary = [
        row
        for row in summary_rows
        if row["representation"] == config.primary_name and row["split"] == config.primary_split
    ]
    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["candidate_rmse_ratio_vs_control_median"]
                        for row in primary
                        if int(row["context_length"]) == context and int(row["horizon"]) == horizon
                    )
                )
                for horizon in config.horizons
            ]
            for context in config.context_lengths
        ]
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.8, vmax=1.2, cmap="coolwarm")
    axis.set_xticks(range(len(config.horizons)), config.horizons)
    axis.set_yticks(range(len(config.context_lengths)), config.context_lengths)
    axis.set_xlabel("Horizon")
    axis.set_ylabel("Context length")
    axis.set_title(f"Candidate vs {config.primary_name}\nrun={run_id}", fontsize=10.0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis, label="Median RMSE ratio")
    figure.savefig(plots_dir / "summary__primary-control-heatmap.png", dpi=config.plot_dpi)
    plt.close(figure)

    controls = gate["controls"]
    if not isinstance(controls, Mapping):
        raise RuntimeError("gate controls must be a mapping")
    predictive = []
    pareto = []
    labels = []
    for control in config.gate_names:
        payload = controls[control]
        if not isinstance(payload, Mapping) or not isinstance(payload["observed"], Mapping):
            raise RuntimeError("gate control observations must be mappings")
        observed = payload["observed"]
        predictive.append(int(observed["robust_predictive_cells"]))
        pareto.append(int(observed["robust_pareto_cells"]))
        labels.append(control.replace("_geometry", "").replace("_", "\n"))
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.bar(positions - 0.18, predictive, width=0.36, label="Predictive")
    axis.bar(positions + 0.18, pareto, width=0.36, label="Pareto")
    axis.axhline(math.ceil(config.minimum_robust_cell_rate * 9), color="black", linestyle="--")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Robust validation cells (of 9)")
    axis.set_title(f"Gate coverage by causal control\nrun={run_id}", fontsize=10.0)
    axis.legend()
    figure.savefig(plots_dir / "summary__gate-cell-coverage.png", dpi=config.plot_dpi)
    plt.close(figure)

    selected = Counter(
        (str(row["control"]), str(row["parameter"]))
        for row in tuning_rows
        if row["selected"] is True
    )
    keys = list(selected)
    figure, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    axis.bar(range(len(keys)), [selected[key] for key in keys])
    axis.set_xticks(
        range(len(keys)),
        [f"{control.replace('_geometry', '')}\n{parameter}" for control, parameter in keys],
        fontsize=7,
    )
    axis.set_ylabel("Selections across seed/cell")
    axis.set_title(f"Train-only tuning selections\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__tuning-selections.png", dpi=config.plot_dpi)
    plt.close(figure)


def _write_partial(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
    tuning_rows: Sequence[Mapping[str, object]],
) -> None:
    forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, rows)
    forecasting._write_csv(
        run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
    )
    forecasting._write_csv(run_dir / "tuning.csv", TUNING_FIELDS, tuning_rows)


def _parameter_candidates(control: str, config: ControlConfig) -> tuple[int | float, ...]:
    if control == "moving_average_geometry":
        return config.moving_average_windows
    if control == "ewma_geometry":
        return config.ewma_alphas
    if control == "fixed_geometry":
        return config.fixed_segment_lengths
    if control == "abba_geometry":
        return config.abba_tolerances
    raise ValueError(f"{control} has no tuning candidates")


def _role(name: str, config: ControlConfig) -> str:
    if name in {"raw", "first_difference"}:
        return "baseline"
    if name == config.candidate_name:
        return "candidate"
    if name == config.primary_name:
        return "primary_control"
    return "external_control" if name == "abba_geometry" else "causal_control"


def _causal_scope(name: str) -> str:
    if name == "abba_geometry":
        return "window_offline"
    if name == "fixed_geometry":
        return "forecast_causal_fixed_boundaries"
    return "strict_online"


def _verify_external_dependencies(config: ControlConfig) -> dict[str, str]:
    if "abba_geometry" not in config.control_names:
        return {}
    try:
        installed = version("fabba")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "abba_geometry requires `uv sync --group abba` or `uv run --group abba ...`"
        ) from error
    if installed != config.fabba_version:
        raise RuntimeError(
            f"abba_geometry requires fabba=={config.fabba_version}, found {installed}"
        )
    return {"fabba": installed}


def _three_increasing_integers(value: object, name: str) -> tuple[int, ...]:
    result = forecasting_robustness._increasing_integer_tuple(value, name, minimum=1)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three candidates")
    return result


def _three_increasing_reals(value: object, name: str) -> tuple[float, ...]:
    result = forecasting_robustness._increasing_real_tuple(value, name, minimum=0.0)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three candidates")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = forecasting._real(value, name, minimum=0.0)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < value <= 1")
    return result


def _format_parameter(parameter: int | float) -> str:
    return str(parameter)


def _values(rows: Sequence[Mapping[str, object]], field: str) -> NDArray[np.float64]:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _optional_values(rows: Sequence[Mapping[str, object]], field: str) -> NDArray[np.float64]:
    return np.asarray(
        [float(row[field]) for row in rows if row.get(field, "") != ""],
        dtype=np.float64,
    )


def _optional_mean(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    values = _optional_values(rows, field)
    return float(np.mean(values)) if values.size else ""


def _boolean_rate(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    values = [bool(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else ""


def _geometric_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(values))))


def _summary_sort_key(row: Mapping[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["representation"]),
        str(row["split"]),
        int(row["horizon"]),
        int(row["context_length"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/controls.toml"),
        help="TOML Stage-8 control configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line Stage-8 control benchmark."""

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
