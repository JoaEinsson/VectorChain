"""Factorial robustness grid for the leakage-safe forecasting benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from experiments import forecasting
else:
    import forecasting

CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    *forecasting.METRIC_FIELDS,
)

SIGNAL_CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    *forecasting.SIGNAL_METRIC_FIELDS,
)

SUMMARY_FIELDS = (
    "representation",
    "split",
    "horizon",
    "context_length",
    "tolerance",
    "n_seeds",
    "n_successes",
    "n_failures",
    "mae_mean",
    "mae_median",
    "rmse_mean",
    "rmse_median",
    "rmse_q1",
    "rmse_q3",
    "rmse_min",
    "rmse_max",
    "rmse_ratio_vs_raw_mean",
    "rmse_ratio_vs_raw_median",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "mean_input_bytes",
    "step_reduction_factor_vs_raw_mean",
    "scalar_reduction_factor_vs_raw_mean",
    "predictive_parity_rate",
    "structural_reduction_success_rate",
    "payload_reduction_success_rate",
    "joint_success_rate",
    "robust_cell",
    "representation_runtime_median_s",
)

SIGNAL_SUMMARY_FIELDS = (
    "representation",
    "split",
    "horizon",
    "context_length",
    "tolerance",
    "signal",
    "n_seeds",
    "mae_mean",
    "mae_median",
    "rmse_mean",
    "rmse_median",
    "rmse_q1",
    "rmse_q3",
    "rmse_min",
    "rmse_max",
)


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Validated grid layered over the canonical forecasting configuration."""

    name: str
    base_config_path: Path
    base: forecasting.ForecastConfig
    seeds: tuple[int, ...]
    context_lengths: tuple[int, ...]
    horizons: tuple[int, ...]
    tolerances: tuple[float, ...]
    stride: int
    predictive_parity_ratio: float
    maximum_step_fraction: float
    maximum_scalar_fraction: float
    robust_seed_rate: float
    repetitions: int
    warmup_repetitions: int
    output_root: str
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of one robustness grid run."""

    run_id: str
    run_dir: Path
    summary_path: Path
    n_conditions: int
    n_failures: int


def load_config(path: Path) -> RobustnessConfig:
    """Load a robustness grid and its frozen base forecasting configuration."""

    path = path.resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = forecasting._table(raw, "experiment")
    grid = forecasting._table(raw, "grid")
    criteria = forecasting._table(raw, "criteria")
    timing = forecasting._table(raw, "timing")
    output = forecasting._table(raw, "output")

    name = forecasting._non_empty_string(experiment.get("name"), "experiment.name")
    base_reference = forecasting._non_empty_string(
        experiment.get("base_config"), "experiment.base_config"
    )
    candidate = Path(base_reference)
    base_config_path = candidate if candidate.is_absolute() else path.parent / candidate
    base_config_path = base_config_path.resolve()
    base = forecasting.load_config(base_config_path)
    seeds = _unique_integer_tuple(experiment.get("seeds"), "experiment.seeds", minimum=0)
    if len(seeds) < 2:
        msg = "experiment.seeds must contain at least two independent seeds"
        raise ValueError(msg)
    context_lengths = _increasing_integer_tuple(
        grid.get("context_lengths"), "grid.context_lengths", minimum=3
    )
    horizons = _increasing_integer_tuple(grid.get("horizons"), "grid.horizons", minimum=1)
    tolerances = _increasing_real_tuple(grid.get("tolerances"), "grid.tolerances", minimum=0.0)
    stride = forecasting._integer(grid.get("stride"), "grid.stride", minimum=1)

    predictive_parity_ratio = forecasting._real(
        criteria.get("predictive_parity_ratio"),
        "criteria.predictive_parity_ratio",
        minimum=1.0,
    )
    maximum_step_fraction = forecasting._real(
        criteria.get("maximum_step_fraction"),
        "criteria.maximum_step_fraction",
        minimum=0.0,
    )
    maximum_scalar_fraction = forecasting._real(
        criteria.get("maximum_scalar_fraction"),
        "criteria.maximum_scalar_fraction",
        minimum=0.0,
    )
    robust_seed_rate = forecasting._real(
        criteria.get("robust_seed_rate"), "criteria.robust_seed_rate", minimum=0.0
    )
    if maximum_step_fraction <= 0.0 or maximum_scalar_fraction <= 0.0:
        msg = "maximum input fractions must be positive"
        raise ValueError(msg)
    if robust_seed_rate <= 0.0 or robust_seed_rate > 1.0:
        msg = "criteria.robust_seed_rate must satisfy 0 < value <= 1"
        raise ValueError(msg)

    repetitions = forecasting._integer(timing.get("repetitions"), "timing.repetitions", minimum=1)
    warmups = forecasting._integer(
        timing.get("warmup_repetitions"), "timing.warmup_repetitions", minimum=0
    )
    output_root = forecasting._non_empty_string(output.get("root"), "output.root")
    save_plots = forecasting._boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = forecasting._integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    if base.context_length not in context_lengths:
        msg = "grid.context_lengths must include the base context length"
        raise ValueError(msg)
    if base.horizon not in horizons:
        msg = "grid.horizons must include the base horizon"
        raise ValueError(msg)
    if base.vectorchain_tolerance not in tolerances:
        msg = "grid.tolerances must include the base VectorChain tolerance"
        raise ValueError(msg)

    config = RobustnessConfig(
        name=name,
        base_config_path=base_config_path,
        base=base,
        seeds=seeds,
        context_lengths=context_lengths,
        horizons=horizons,
        tolerances=tolerances,
        stride=stride,
        predictive_parity_ratio=predictive_parity_ratio,
        maximum_step_fraction=maximum_step_fraction,
        maximum_scalar_fraction=maximum_scalar_fraction,
        robust_seed_rate=robust_seed_rate,
        repetitions=repetitions,
        warmup_repetitions=warmups,
        output_root=output_root,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )
    for context_length in config.context_lengths:
        for horizon in config.horizons:
            condition = _condition_config(config, config.seeds[0], context_length, horizon)
            forecasting._validate_split_counts(condition)
    return config


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute and aggregate the complete registered robustness grid."""

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
        * len(config.context_lengths)
        * len(config.horizons)
        * (2 + len(config.tolerances))
    )
    effective_config = dict(config.raw)
    effective_config["base_forecasting_config"] = config.base.raw
    effective_config["resolved"] = {
        "config_path": forecasting._display_path(config_path, repository_root),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "base_config_path": forecasting._display_path(config.base_config_path, repository_root),
        "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "combined_sha256": config_digest,
        "output_root": str(resolved_root),
        "expected_condition_evaluations": expected_conditions,
    }
    forecasting._write_json(run_dir / "config.json", effective_config)
    environment = _environment_manifest(
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=effective_config,
        command_args=tuple(command_args if command_args is not None else sys.argv),
    )
    forecasting._write_json(run_dir / "environment.json", environment)

    condition_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
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
            for context_length in config.context_lengths:
                for horizon in config.horizons:
                    common = replace(
                        seed_config,
                        context_length=context_length,
                        horizon=horizon,
                    )
                    examples = forecasting._build_examples(signals, signal_seeds, common)
                    for representation in ("raw", "first_difference"):
                        _evaluate_condition(
                            representation,
                            None,
                            seed,
                            context_length,
                            horizon,
                            examples,
                            common,
                            condition_rows,
                            signal_rows,
                        )
                        _write_partial(run_dir, condition_rows, signal_rows)
                    for tolerance in config.tolerances:
                        vector_config = replace(common, vectorchain_tolerance=tolerance)
                        _evaluate_condition(
                            "vectorchain",
                            tolerance,
                            seed,
                            context_length,
                            horizon,
                            examples,
                            vector_config,
                            condition_rows,
                            signal_rows,
                        )
                        _write_partial(run_dir, condition_rows, signal_rows)

        _add_paired_comparisons(condition_rows, config)
        summary_rows = _summarize_conditions(condition_rows, config)
        signal_summary_rows = _summarize_signals(signal_rows)
        forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
        forecasting._write_csv(
            run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
        )
        forecasting._write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
        forecasting._write_csv(
            run_dir / "summary_by_signal.csv", SIGNAL_SUMMARY_FIELDS, signal_summary_rows
        )
        if config.save_plots and all(row["status"] == "ok" for row in condition_rows):
            _plot_summaries(condition_rows, summary_rows, config, plots_dir, run_id)
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
        environment["derived_signal_seeds"] = derived_seeds
        forecasting._write_json(run_dir / "environment.json", environment)
        forecasting._write_manifest(run_dir, status="failed")
        raise

    failed_conditions = {
        str(row["condition_id"]) for row in condition_rows if row["status"] != "ok"
    }
    environment["status"] = "complete" if not failed_conditions else "complete_with_failures"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
    environment["n_conditions"] = expected_conditions
    environment["n_failures"] = len(failed_conditions)
    environment["derived_signal_seeds"] = derived_seeds
    forecasting._write_json(run_dir / "environment.json", environment)
    forecasting._write_manifest(run_dir, status=str(environment["status"]))
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        summary_path=run_dir / "summary.csv",
        n_conditions=expected_conditions,
        n_failures=len(failed_conditions),
    )


def _condition_config(
    config: RobustnessConfig, seed: int, context_length: int, horizon: int
) -> forecasting.ForecastConfig:
    return replace(
        config.base,
        seed=seed,
        context_length=context_length,
        horizon=horizon,
        stride=config.stride,
        repetitions=config.repetitions,
        warmup_repetitions=config.warmup_repetitions,
        save_models=False,
        save_plots=False,
    )


def _evaluate_condition(
    representation: str,
    tolerance: float | None,
    seed: int,
    context_length: int,
    horizon: int,
    examples: Sequence[forecasting.ForecastExample],
    condition_config: forecasting.ForecastConfig,
    condition_rows: list[dict[str, object]],
    signal_rows: list[dict[str, object]],
) -> None:
    condition_id = _condition_id(representation, seed, context_length, horizon, tolerance)
    prefix: dict[str, object] = {
        "condition_id": condition_id,
        "seed": seed,
        "horizon": horizon,
        "context_length": context_length,
        "tolerance": "" if tolerance is None else tolerance,
    }
    try:
        result = forecasting._run_representation(representation, examples, condition_config)
        condition_rows.extend({**prefix, **row} for row in result[0])
        signal_rows.extend({**prefix, **row} for row in result[1])
    except Exception as error:
        condition_rows.extend(
            {
                **prefix,
                **forecasting._failure_row(representation, split, error),
            }
            for split in ("validation", "test")
        )


def _condition_id(
    representation: str,
    seed: int,
    context_length: int,
    horizon: int,
    tolerance: float | None,
) -> str:
    tolerance_tag = "na" if tolerance is None else _number_tag(tolerance)
    return f"seed-{seed}__h-{horizon}__c-{context_length}__tol-{tolerance_tag}__{representation}"


def _write_partial(
    run_dir: Path,
    condition_rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
) -> None:
    forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
    forecasting._write_csv(
        run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
    )


def _add_paired_comparisons(rows: Sequence[dict[str, object]], config: RobustnessConfig) -> None:
    raw_by_key = {
        (
            int(row["seed"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["split"]),
        ): row
        for row in rows
        if row["representation"] == "raw" and row["status"] == "ok"
    }
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["split"]),
        )
        if row["status"] != "ok" or key not in raw_by_key:
            continue
        raw = raw_by_key[key]
        rmse_ratio = float(row["rmse"]) / float(raw["rmse"])
        step_factor = float(raw["mean_input_steps"]) / float(row["mean_input_steps"])
        scalar_factor = float(raw["mean_input_scalar_elements"]) / float(
            row["mean_input_scalar_elements"]
        )
        row["rmse_ratio_vs_raw"] = rmse_ratio
        row["step_reduction_factor_vs_raw"] = step_factor
        row["scalar_reduction_factor_vs_raw"] = scalar_factor
        if row["representation"] == "vectorchain":
            predictive = rmse_ratio <= config.predictive_parity_ratio
            structural = float(row["mean_input_steps"]) <= (
                config.maximum_step_fraction * float(raw["mean_input_steps"])
            )
            payload = float(row["mean_input_scalar_elements"]) <= (
                config.maximum_scalar_fraction * float(raw["mean_input_scalar_elements"])
            )
            row["predictive_parity_vs_raw"] = predictive
            row["structural_reduction_success"] = structural
            row["payload_reduction_success"] = payload
            row["joint_success"] = predictive and structural and payload


def _summarize_conditions(
    rows: Sequence[Mapping[str, object]], config: RobustnessConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["split"],
            row["horizon"],
            row["context_length"],
            row["tolerance"],
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        successful = [row for row in group if row["status"] == "ok"]
        row: dict[str, object] = {
            "representation": key[0],
            "split": key[1],
            "horizon": key[2],
            "context_length": key[3],
            "tolerance": key[4],
            "n_seeds": len({int(item["seed"]) for item in group}),
            "n_successes": len(successful),
            "n_failures": len(group) - len(successful),
        }
        if successful:
            mae = _values(successful, "mae")
            rmse = _values(successful, "rmse")
            ratio = _optional_values(successful, "rmse_ratio_vs_raw")
            step_factors = _optional_values(successful, "step_reduction_factor_vs_raw")
            scalar_factors = _optional_values(successful, "scalar_reduction_factor_vs_raw")
            row.update(
                {
                    "mae_mean": float(np.mean(mae)),
                    "mae_median": float(np.median(mae)),
                    "rmse_mean": float(np.mean(rmse)),
                    "rmse_median": float(np.median(rmse)),
                    "rmse_q1": float(np.quantile(rmse, 0.25)),
                    "rmse_q3": float(np.quantile(rmse, 0.75)),
                    "rmse_min": float(np.min(rmse)),
                    "rmse_max": float(np.max(rmse)),
                    "rmse_ratio_vs_raw_mean": (float(np.mean(ratio)) if ratio.size else ""),
                    "rmse_ratio_vs_raw_median": (float(np.median(ratio)) if ratio.size else ""),
                    "mean_input_steps": float(np.mean(_values(successful, "mean_input_steps"))),
                    "mean_input_scalar_elements": float(
                        np.mean(_values(successful, "mean_input_scalar_elements"))
                    ),
                    "mean_input_bytes": float(np.mean(_values(successful, "mean_input_bytes"))),
                    "step_reduction_factor_vs_raw_mean": (
                        float(np.mean(step_factors)) if step_factors.size else ""
                    ),
                    "scalar_reduction_factor_vs_raw_mean": (
                        float(np.mean(scalar_factors)) if scalar_factors.size else ""
                    ),
                    "representation_runtime_median_s": float(
                        np.median(_values(successful, "representation_runtime_s"))
                    ),
                }
            )
            if key[0] == "vectorchain":
                comparable = [item for item in successful if item["joint_success"] != ""]
                if comparable:
                    rates = {
                        "predictive_parity_rate": _boolean_rate(
                            comparable, "predictive_parity_vs_raw"
                        ),
                        "structural_reduction_success_rate": _boolean_rate(
                            comparable, "structural_reduction_success"
                        ),
                        "payload_reduction_success_rate": _boolean_rate(
                            comparable, "payload_reduction_success"
                        ),
                        "joint_success_rate": _boolean_rate(comparable, "joint_success"),
                    }
                    row.update(rates)
                    row["robust_cell"] = rates["joint_success_rate"] >= config.robust_seed_rate
        summaries.append({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return sorted(
        summaries,
        key=lambda row: (
            str(row["representation"]),
            str(row["split"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["tolerance"]),
        ),
    )


def _summarize_signals(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["split"],
            row["horizon"],
            row["context_length"],
            row["tolerance"],
            row["signal"],
        )
        grouped[key].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        mae = _values(group, "mae")
        rmse = _values(group, "rmse")
        row: dict[str, object] = {
            "representation": key[0],
            "split": key[1],
            "horizon": key[2],
            "context_length": key[3],
            "tolerance": key[4],
            "signal": key[5],
            "n_seeds": len({int(item["seed"]) for item in group}),
            "mae_mean": float(np.mean(mae)),
            "mae_median": float(np.median(mae)),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_median": float(np.median(rmse)),
            "rmse_q1": float(np.quantile(rmse, 0.25)),
            "rmse_q3": float(np.quantile(rmse, 0.75)),
            "rmse_min": float(np.min(rmse)),
            "rmse_max": float(np.max(rmse)),
        }
        summaries.append(row)
    return sorted(
        summaries,
        key=lambda row: (
            str(row["representation"]),
            str(row["split"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["tolerance"]),
            str(row["signal"]),
        ),
    )


def _values(rows: Sequence[Mapping[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _optional_values(rows: Sequence[Mapping[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows if row[field] != ""], dtype=np.float64)


def _boolean_rate(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows]))


def _plot_summaries(
    condition_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    config: RobustnessConfig,
    plots_dir: Path,
    run_id: str,
) -> None:
    import matplotlib.pyplot as plt

    test_vector = [
        row
        for row in summary_rows
        if row["representation"] == "vectorchain" and row["split"] == "test" and row["n_successes"]
    ]
    figure, axes = plt.subplots(
        1, len(config.tolerances), figsize=(13.0, 4.5), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, tolerance in zip(axes_array, config.tolerances, strict=True):
        matrix = np.asarray(
            [
                [
                    float(
                        next(
                            row["joint_success_rate"]
                            for row in test_vector
                            if float(row["tolerance"]) == tolerance
                            and int(row["context_length"]) == context
                            and int(row["horizon"]) == horizon
                        )
                    )
                    for horizon in config.horizons
                ]
                for context in config.context_lengths
            ]
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(len(config.horizons)), config.horizons)
        axis.set_yticks(range(len(config.context_lengths)), config.context_lengths)
        axis.set_xlabel("Horizon")
        axis.set_ylabel("Context length")
        axis.set_title(f"Tolerance={tolerance:g}")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < 0.6 else "black",
                )
    figure.colorbar(image, ax=axes_array.tolist(), label="Joint success rate")
    figure.suptitle(f"VectorChain robustness across five seeds\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__joint-success-rate.png", dpi=config.plot_dpi)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(config.tolerances), figsize=(13.0, 4.5), constrained_layout=True, sharey=True
    )
    axes_array = np.atleast_1d(axes)
    for axis, tolerance in zip(axes_array, config.tolerances, strict=True):
        for context in config.context_lengths:
            rows = sorted(
                (
                    row
                    for row in test_vector
                    if float(row["tolerance"]) == tolerance
                    and int(row["context_length"]) == context
                ),
                key=lambda row: int(row["horizon"]),
            )
            axis.plot(
                [int(row["horizon"]) for row in rows],
                [float(row["rmse_ratio_vs_raw_mean"]) for row in rows],
                marker="o",
                label=f"context={context}",
            )
        axis.axhline(config.predictive_parity_ratio, color="black", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(config.horizons, config.horizons)
        axis.set_xlabel("Horizon")
        axis.set_title(f"Tolerance={tolerance:g}")
        axis.grid(alpha=0.25)
    axes_array[0].set_ylabel("Mean RMSE ratio vs raw")
    axes_array[-1].legend(fontsize="small")
    figure.suptitle(f"Predictive parity sensitivity\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__rmse-ratio.png", dpi=config.plot_dpi)
    plt.close(figure)

    successful_vector = [
        row
        for row in condition_rows
        if row["representation"] == "vectorchain"
        and row["split"] == "test"
        and row["status"] == "ok"
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for tolerance in config.tolerances:
        rows = [row for row in successful_vector if float(row["tolerance"]) == tolerance]
        axis.scatter(
            [float(row["scalar_reduction_factor_vs_raw"]) for row in rows],
            [float(row["rmse_ratio_vs_raw"]) for row in rows],
            alpha=0.7,
            label=f"tolerance={tolerance:g}",
        )
    axis.axhline(config.predictive_parity_ratio, color="black", linestyle="--")
    axis.axvline(1.0, color="black", linestyle=":")
    axis.set_xlabel("Scalar reduction factor vs raw")
    axis.set_ylabel("RMSE ratio vs raw")
    axis.set_title(f"Payload-prediction tradeoff by seed and cell\nrun={run_id}", fontsize=10.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__payload-parity-tradeoff.png", dpi=config.plot_dpi)
    plt.close(figure)

    first_difference = [
        float(row["rmse_ratio_vs_raw"])
        for row in condition_rows
        if row["representation"] == "first_difference"
        and row["split"] == "test"
        and row["status"] == "ok"
    ]
    distributions = [first_difference]
    labels = ["First difference"]
    for tolerance in config.tolerances:
        distributions.append(
            [
                float(row["rmse_ratio_vs_raw"])
                for row in successful_vector
                if float(row["tolerance"]) == tolerance
            ]
        )
        labels.append(f"VC {tolerance:g}")
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    axis.boxplot(distributions, tick_labels=labels, showmeans=True)
    axis.axhline(config.predictive_parity_ratio, color="black", linestyle="--")
    axis.set_ylabel("Test RMSE ratio vs paired raw")
    axis.set_title(f"Distribution over seeds and grid cells\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(plots_dir / "summary__ratio-distributions.png", dpi=config.plot_dpi)
    plt.close(figure)


def _environment_manifest(
    *,
    run_id: str,
    started: datetime,
    git_commit: str,
    git_dirty: bool,
    config: Mapping[str, Any],
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
        "command": {"argv": list(command_args), "display": " ".join(command_args)},
    }


def _unique_integer_tuple(value: object, name: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty list of integers"
        raise ValueError(msg)
    result = tuple(forecasting._integer(item, name, minimum=minimum) for item in value)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicates"
        raise ValueError(msg)
    return result


def _increasing_integer_tuple(value: object, name: str, *, minimum: int) -> tuple[int, ...]:
    result = _unique_integer_tuple(value, name, minimum=minimum)
    if any(right <= left for left, right in itertools.pairwise(result)):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)
    return result


def _increasing_real_tuple(value: object, name: str, *, minimum: float) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty list of numbers"
        raise ValueError(msg)
    result = tuple(forecasting._real(item, name, minimum=minimum) for item in value)
    if any(right <= left for left, right in itertools.pairwise(result)):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)
    return result


def _number_tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/robustness.toml"),
        help="TOML forecasting robustness grid",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line robustness grid entry point."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("05_forecasting_robustness.py", *argv))
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
