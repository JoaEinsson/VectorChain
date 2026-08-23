"""Forecasting ablation of absolute and relational VectorChain features."""

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

from vectorchain.features import FeatureName, validate_feature_names

if __package__:
    from experiments import forecasting, forecasting_robustness
else:
    import forecasting
    import forecasting_robustness

CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    "variant",
    "features",
    "role",
    *forecasting.METRIC_FIELDS,
    "rmse_ratio_vs_reference",
    "beats_reference",
    "practical_improvement_vs_reference",
)

SIGNAL_CONDITION_FIELDS = (
    "condition_id",
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    "variant",
    "features",
    "role",
    *forecasting.SIGNAL_METRIC_FIELDS,
    "rmse_ratio_vs_reference",
)

STEP_AUDIT_FIELDS = (
    "seed",
    "horizon",
    "context_length",
    "tolerance",
    "variant",
    "features",
    "n_examples",
    "step_signature_sha256",
    "matches_reference",
)

SUMMARY_FIELDS = (
    "representation",
    "variant",
    "features",
    "role",
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
    "rmse_ratio_vs_reference_mean",
    "rmse_ratio_vs_reference_median",
    "mean_input_steps",
    "mean_input_scalar_elements",
    "mean_input_bytes",
    "n_pooled_features",
    "n_model_parameters",
    "step_reduction_factor_vs_raw_mean",
    "scalar_reduction_factor_vs_raw_mean",
    "predictive_parity_rate",
    "structural_reduction_success_rate",
    "payload_reduction_success_rate",
    "joint_success_rate",
    "beats_reference_rate",
    "practical_improvement_rate",
    "robust_improvement_cell",
    "representation_runtime_median_s",
)

SEED_SUMMARY_FIELDS = (
    "variant",
    "features",
    "role",
    "split",
    "seed",
    "n_cells",
    "geometric_mean_rmse_ratio_vs_reference",
    "rmse_ratio_vs_raw_mean",
    "primary_seed_success",
    "capacity_control_seed_success",
)

SIGNAL_SUMMARY_FIELDS = (
    "representation",
    "variant",
    "features",
    "role",
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
    "rmse_ratio_vs_reference_mean",
    "rmse_ratio_vs_reference_median",
)


@dataclass(frozen=True, slots=True)
class FeatureAblationConfig:
    """Validated feature-ablation program layered over forecasting."""

    name: str
    base_config_path: Path
    base: forecasting.ForecastConfig
    seeds: tuple[int, ...]
    context_lengths: tuple[int, ...]
    horizons: tuple[int, ...]
    stride: int
    tolerance: float
    variant_names: tuple[str, ...]
    variant_features: Mapping[str, tuple[FeatureName, ...]]
    reference_variant: str
    primary_variant: str
    capacity_control_variant: str
    primary_split: str
    maximum_primary_rmse_ratio: float
    maximum_capacity_control_rmse_ratio: float
    robust_seed_rate: float
    minimum_robust_cell_rate: float
    predictive_parity_ratio: float
    maximum_step_fraction: float
    maximum_scalar_fraction: float
    repetitions: int
    warmup_repetitions: int
    output_root: str
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of one feature-ablation run."""

    run_id: str
    run_dir: Path
    summary_path: Path
    gate_path: Path
    n_conditions: int
    n_failures: int
    gate_passed: bool


def load_config(path: Path) -> FeatureAblationConfig:
    """Load and validate the registered forecasting feature ablation."""

    path = path.resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = forecasting._table(raw, "experiment")
    grid = forecasting._table(raw, "grid")
    ablations = forecasting._table(raw, "ablations")
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

    seeds = forecasting_robustness._unique_integer_tuple(
        experiment.get("seeds"), "experiment.seeds", minimum=0
    )
    if len(seeds) < 2:
        msg = "experiment.seeds must contain at least two independent seeds"
        raise ValueError(msg)
    context_lengths = forecasting_robustness._increasing_integer_tuple(
        grid.get("context_lengths"), "grid.context_lengths", minimum=3
    )
    horizons = forecasting_robustness._increasing_integer_tuple(
        grid.get("horizons"), "grid.horizons", minimum=1
    )
    stride = forecasting._integer(grid.get("stride"), "grid.stride", minimum=1)
    tolerance = forecasting._real(grid.get("tolerance"), "grid.tolerance", minimum=0.0)

    variant_names = forecasting._unique_string_tuple(ablations.get("names"), "ablations.names")
    raw_features = forecasting._table(ablations, "features")
    if set(raw_features) != set(variant_names):
        msg = "ablations.features must contain exactly one entry for every variant"
        raise ValueError(msg)
    variant_features: dict[str, tuple[FeatureName, ...]] = {}
    for variant in variant_names:
        value = raw_features[variant]
        if not isinstance(value, list):
            msg = f"ablations.features.{variant} must be a list"
            raise TypeError(msg)
        variant_features[variant] = validate_feature_names(value)

    reference_variant = _configured_variant(ablations, "reference_variant", variant_names)
    primary_variant = _configured_variant(ablations, "primary_variant", variant_names)
    capacity_variant = _configured_variant(ablations, "capacity_control_variant", variant_names)
    if len({reference_variant, primary_variant, capacity_variant}) != 3:
        msg = "reference, primary, and capacity-control variants must be distinct"
        raise ValueError(msg)
    if len(variant_features[reference_variant]) != len(variant_features[capacity_variant]):
        msg = "capacity-control and reference variants must have the same feature count"
        raise ValueError(msg)

    primary_split = forecasting._non_empty_string(
        criteria.get("primary_split"), "criteria.primary_split"
    )
    if primary_split != "validation":
        msg = "criteria.primary_split must be 'validation' for this observed benchmark"
        raise ValueError(msg)
    maximum_primary_ratio = forecasting._real(
        criteria.get("maximum_primary_rmse_ratio"),
        "criteria.maximum_primary_rmse_ratio",
        minimum=0.0,
    )
    maximum_capacity_ratio = forecasting._real(
        criteria.get("maximum_capacity_control_rmse_ratio"),
        "criteria.maximum_capacity_control_rmse_ratio",
        minimum=0.0,
    )
    if maximum_primary_ratio <= 0.0 or maximum_primary_ratio >= 1.0:
        msg = "criteria.maximum_primary_rmse_ratio must satisfy 0 < value < 1"
        raise ValueError(msg)
    if maximum_capacity_ratio <= 0.0:
        msg = "criteria.maximum_capacity_control_rmse_ratio must be positive"
        raise ValueError(msg)
    robust_seed_rate = _unit_interval(
        criteria.get("robust_seed_rate"), "criteria.robust_seed_rate", exclude_zero=True
    )
    minimum_robust_cell_rate = _unit_interval(
        criteria.get("minimum_robust_cell_rate"),
        "criteria.minimum_robust_cell_rate",
        exclude_zero=True,
    )
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
    if maximum_step_fraction <= 0.0 or maximum_scalar_fraction <= 0.0:
        msg = "maximum input fractions must be positive"
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

    config = FeatureAblationConfig(
        name=name,
        base_config_path=base_config_path,
        base=base,
        seeds=seeds,
        context_lengths=context_lengths,
        horizons=horizons,
        stride=stride,
        tolerance=tolerance,
        variant_names=variant_names,
        variant_features=variant_features,
        reference_variant=reference_variant,
        primary_variant=primary_variant,
        capacity_control_variant=capacity_variant,
        primary_split=primary_split,
        maximum_primary_rmse_ratio=maximum_primary_ratio,
        maximum_capacity_control_rmse_ratio=maximum_capacity_ratio,
        robust_seed_rate=robust_seed_rate,
        minimum_robust_cell_rate=minimum_robust_cell_rate,
        predictive_parity_ratio=predictive_parity_ratio,
        maximum_step_fraction=maximum_step_fraction,
        maximum_scalar_fraction=maximum_scalar_fraction,
        repetitions=repetitions,
        warmup_repetitions=warmups,
        output_root=output_root,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )
    for context_length in config.context_lengths:
        for horizon in config.horizons:
            forecasting._validate_split_counts(
                _condition_config(config, config.seeds[0], context_length, horizon)
            )
    return config


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute the complete registered feature-ablation grid."""

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
        * (2 + len(config.variant_names))
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
    environment = forecasting_robustness._environment_manifest(
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
    step_audit_rows: list[dict[str, object]] = []
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
                        vectorchain_tolerance=config.tolerance,
                    )
                    examples = forecasting._build_examples(signals, signal_seeds, common)
                    for representation in ("raw", "first_difference"):
                        _evaluate_condition(
                            representation=representation,
                            variant="",
                            features=(),
                            seed=seed,
                            context_length=context_length,
                            horizon=horizon,
                            examples=examples,
                            condition_config=common,
                            config=config,
                            condition_rows=condition_rows,
                            signal_rows=signal_rows,
                        )
                        _write_partial(run_dir, condition_rows, signal_rows, step_audit_rows)

                    input_rows_by_variant: dict[str, list[dict[str, object]]] = {}
                    for variant in config.variant_names:
                        features = config.variant_features[variant]
                        variant_config = replace(common, vectorchain_features=features)
                        input_rows = _evaluate_condition(
                            representation="vectorchain",
                            variant=variant,
                            features=features,
                            seed=seed,
                            context_length=context_length,
                            horizon=horizon,
                            examples=examples,
                            condition_config=variant_config,
                            config=config,
                            condition_rows=condition_rows,
                            signal_rows=signal_rows,
                        )
                        if input_rows:
                            input_rows_by_variant[variant] = input_rows
                        _write_partial(run_dir, condition_rows, signal_rows, step_audit_rows)
                    step_audit_rows.extend(
                        _audit_steps(
                            input_rows_by_variant,
                            config,
                            seed=seed,
                            context_length=context_length,
                            horizon=horizon,
                        )
                    )
                    _write_partial(run_dir, condition_rows, signal_rows, step_audit_rows)

        _add_paired_comparisons(condition_rows, signal_rows, config)
        summary_rows = _summarize_conditions(condition_rows, config)
        seed_summary_rows = _summarize_seeds(condition_rows, config)
        signal_summary_rows = _summarize_signals(signal_rows)
        gate = _evaluate_gate(summary_rows, seed_summary_rows, config)
        forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
        forecasting._write_csv(
            run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
        )
        forecasting._write_csv(run_dir / "step_audit.csv", STEP_AUDIT_FIELDS, step_audit_rows)
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
                condition_rows, summary_rows, seed_summary_rows, config, plots_dir, run_id
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

    failed_conditions = {
        str(row["condition_id"]) for row in condition_rows if row["status"] != "ok"
    }
    environment["status"] = "complete" if not failed_conditions else "complete_with_failures"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
    environment["n_conditions"] = expected_conditions
    environment["n_failures"] = len(failed_conditions)
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
        n_failures=len(failed_conditions),
        gate_passed=bool(gate["passed"]),
    )


def _condition_config(
    config: FeatureAblationConfig, seed: int, context_length: int, horizon: int
) -> forecasting.ForecastConfig:
    return replace(
        config.base,
        seed=seed,
        context_length=context_length,
        horizon=horizon,
        stride=config.stride,
        vectorchain_tolerance=config.tolerance,
        repetitions=config.repetitions,
        warmup_repetitions=config.warmup_repetitions,
        save_models=False,
        save_plots=False,
    )


def _evaluate_condition(
    *,
    representation: str,
    variant: str,
    features: Sequence[FeatureName],
    seed: int,
    context_length: int,
    horizon: int,
    examples: Sequence[forecasting.ForecastExample],
    condition_config: forecasting.ForecastConfig,
    config: FeatureAblationConfig,
    condition_rows: list[dict[str, object]],
    signal_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    condition_id = _condition_id(representation, variant, seed, context_length, horizon)
    prefix: dict[str, object] = {
        "condition_id": condition_id,
        "seed": seed,
        "horizon": horizon,
        "context_length": context_length,
        "tolerance": config.tolerance if representation == "vectorchain" else "",
        "variant": variant,
        "features": ",".join(features),
        "role": _role(representation, variant, config),
    }
    try:
        result = forecasting._run_representation(representation, examples, condition_config)
        condition_rows.extend({**prefix, **row} for row in result[0])
        signal_rows.extend({**prefix, **row} for row in result[1])
        return result[2]
    except Exception as error:
        condition_rows.extend(
            {
                **prefix,
                **forecasting._failure_row(representation, split, error),
            }
            for split in ("validation", "test")
        )
        return []


def _audit_steps(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
    config: FeatureAblationConfig,
    *,
    seed: int,
    context_length: int,
    horizon: int,
) -> list[dict[str, object]]:
    if config.reference_variant not in rows_by_variant:
        return []
    reference = _steps_by_example(rows_by_variant[config.reference_variant])
    audit_rows: list[dict[str, object]] = []
    for variant in config.variant_names:
        if variant not in rows_by_variant:
            continue
        steps = _steps_by_example(rows_by_variant[variant])
        matches = steps == reference
        if not matches:
            msg = f"input steps changed between {config.reference_variant} and {variant}"
            raise RuntimeError(msg)
        payload = "\n".join(f"{key}:{steps[key]}" for key in sorted(steps)).encode()
        audit_rows.append(
            {
                "seed": seed,
                "horizon": horizon,
                "context_length": context_length,
                "tolerance": config.tolerance,
                "variant": variant,
                "features": ",".join(config.variant_features[variant]),
                "n_examples": len(steps),
                "step_signature_sha256": hashlib.sha256(payload).hexdigest(),
                "matches_reference": matches,
            }
        )
    return audit_rows


def _steps_by_example(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    result = {str(row["example_id"]): int(row["input_steps"]) for row in rows}
    if len(result) != len(rows):
        msg = "input rows must contain unique example IDs"
        raise RuntimeError(msg)
    return result


def _add_paired_comparisons(
    rows: Sequence[dict[str, object]],
    signal_rows: Sequence[dict[str, object]],
    config: FeatureAblationConfig,
) -> None:
    raw_by_key = {
        _condition_key(row): row
        for row in rows
        if row["representation"] == "raw" and row["status"] == "ok"
    }
    reference_by_key = {
        _condition_key(row): row
        for row in rows
        if row["variant"] == config.reference_variant and row["status"] == "ok"
    }
    for row in rows:
        key = _condition_key(row)
        if row["status"] != "ok":
            continue
        if key in raw_by_key:
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
        if row["representation"] == "vectorchain" and key in reference_by_key:
            ratio = float(row["rmse"]) / float(reference_by_key[key]["rmse"])
            row["rmse_ratio_vs_reference"] = ratio
            row["beats_reference"] = ratio < 1.0
            row["practical_improvement_vs_reference"] = ratio <= config.maximum_primary_rmse_ratio

    signal_reference = {
        _signal_key(row): row for row in signal_rows if row["variant"] == config.reference_variant
    }
    for row in signal_rows:
        key = _signal_key(row)
        if row["representation"] == "vectorchain" and key in signal_reference:
            row["rmse_ratio_vs_reference"] = float(row["rmse"]) / float(
                signal_reference[key]["rmse"]
            )


def _summarize_conditions(
    rows: Sequence[Mapping[str, object]], config: FeatureAblationConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["variant"],
            row["features"],
            row["role"],
            row["split"],
            row["horizon"],
            row["context_length"],
            row["tolerance"],
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        successful = [row for row in group if row["status"] == "ok"]
        result: dict[str, object] = {
            "representation": key[0],
            "variant": key[1],
            "features": key[2],
            "role": key[3],
            "split": key[4],
            "horizon": key[5],
            "context_length": key[6],
            "tolerance": key[7],
            "n_seeds": len({int(item["seed"]) for item in group}),
            "n_successes": len(successful),
            "n_failures": len(group) - len(successful),
        }
        if successful:
            mae = _values(successful, "mae")
            rmse = _values(successful, "rmse")
            raw_ratio = _optional_values(successful, "rmse_ratio_vs_raw")
            reference_ratio = _optional_values(successful, "rmse_ratio_vs_reference")
            result.update(
                {
                    "mae_mean": float(np.mean(mae)),
                    "mae_median": float(np.median(mae)),
                    "rmse_mean": float(np.mean(rmse)),
                    "rmse_median": float(np.median(rmse)),
                    "rmse_q1": float(np.quantile(rmse, 0.25)),
                    "rmse_q3": float(np.quantile(rmse, 0.75)),
                    "rmse_min": float(np.min(rmse)),
                    "rmse_max": float(np.max(rmse)),
                    "rmse_ratio_vs_raw_mean": (float(np.mean(raw_ratio)) if raw_ratio.size else ""),
                    "rmse_ratio_vs_raw_median": (
                        float(np.median(raw_ratio)) if raw_ratio.size else ""
                    ),
                    "rmse_ratio_vs_reference_mean": (
                        float(np.mean(reference_ratio)) if reference_ratio.size else ""
                    ),
                    "rmse_ratio_vs_reference_median": (
                        float(np.median(reference_ratio)) if reference_ratio.size else ""
                    ),
                    "mean_input_steps": float(np.mean(_values(successful, "mean_input_steps"))),
                    "mean_input_scalar_elements": float(
                        np.mean(_values(successful, "mean_input_scalar_elements"))
                    ),
                    "mean_input_bytes": float(np.mean(_values(successful, "mean_input_bytes"))),
                    "n_pooled_features": int(successful[0]["n_pooled_features"]),
                    "n_model_parameters": int(successful[0]["n_model_parameters"]),
                    "step_reduction_factor_vs_raw_mean": _optional_mean(
                        successful, "step_reduction_factor_vs_raw"
                    ),
                    "scalar_reduction_factor_vs_raw_mean": _optional_mean(
                        successful, "scalar_reduction_factor_vs_raw"
                    ),
                    "representation_runtime_median_s": float(
                        np.median(_values(successful, "representation_runtime_s"))
                    ),
                }
            )
            if key[0] == "vectorchain":
                practical_rate = _boolean_rate(successful, "practical_improvement_vs_reference")
                result.update(
                    {
                        "predictive_parity_rate": _boolean_rate(
                            successful, "predictive_parity_vs_raw"
                        ),
                        "structural_reduction_success_rate": _boolean_rate(
                            successful, "structural_reduction_success"
                        ),
                        "payload_reduction_success_rate": _boolean_rate(
                            successful, "payload_reduction_success"
                        ),
                        "joint_success_rate": _boolean_rate(successful, "joint_success"),
                        "beats_reference_rate": _boolean_rate(successful, "beats_reference"),
                        "practical_improvement_rate": practical_rate,
                    }
                )
                result["robust_improvement_cell"] = (
                    practical_rate != "" and float(practical_rate) >= config.robust_seed_rate
                )
        summaries.append({field: result.get(field, "") for field in SUMMARY_FIELDS})
    return sorted(summaries, key=_summary_sort_key)


def _summarize_seeds(
    rows: Sequence[Mapping[str, object]], config: FeatureAblationConfig
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row["representation"] != "vectorchain"
            or row["status"] != "ok"
            or row["rmse_ratio_vs_reference"] == ""
        ):
            continue
        key = (
            str(row["variant"]),
            str(row["features"]),
            str(row["role"]),
            str(row["split"]),
            int(row["seed"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    expected_cells = len(config.context_lengths) * len(config.horizons)
    for key, group in grouped.items():
        reference_ratios = _values(group, "rmse_ratio_vs_reference")
        geometric = _geometric_mean(reference_ratios)
        complete = len(group) == expected_cells
        result: dict[str, object] = {
            "variant": key[0],
            "features": key[1],
            "role": key[2],
            "split": key[3],
            "seed": key[4],
            "n_cells": len(group),
            "geometric_mean_rmse_ratio_vs_reference": geometric,
            "rmse_ratio_vs_raw_mean": _optional_mean(group, "rmse_ratio_vs_raw"),
            "primary_seed_success": (
                complete and geometric <= config.maximum_primary_rmse_ratio
                if key[0] == config.primary_variant
                else ""
            ),
            "capacity_control_seed_success": (
                complete and geometric <= config.maximum_capacity_control_rmse_ratio
                if key[0] == config.capacity_control_variant
                else ""
            ),
        }
        summaries.append(result)
    return sorted(
        summaries,
        key=lambda row: (str(row["variant"]), str(row["split"]), int(row["seed"])),
    )


def _summarize_signals(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["representation"],
            row["variant"],
            row["features"],
            row["role"],
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
        ratios = _optional_values(group, "rmse_ratio_vs_reference")
        result: dict[str, object] = {
            "representation": key[0],
            "variant": key[1],
            "features": key[2],
            "role": key[3],
            "split": key[4],
            "horizon": key[5],
            "context_length": key[6],
            "tolerance": key[7],
            "signal": key[8],
            "n_seeds": len({int(item["seed"]) for item in group}),
            "mae_mean": float(np.mean(mae)),
            "mae_median": float(np.median(mae)),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_median": float(np.median(rmse)),
            "rmse_q1": float(np.quantile(rmse, 0.25)),
            "rmse_q3": float(np.quantile(rmse, 0.75)),
            "rmse_min": float(np.min(rmse)),
            "rmse_max": float(np.max(rmse)),
            "rmse_ratio_vs_reference_mean": float(np.mean(ratios)) if ratios.size else "",
            "rmse_ratio_vs_reference_median": (float(np.median(ratios)) if ratios.size else ""),
        }
        summaries.append(result)
    return sorted(
        summaries,
        key=lambda row: (
            str(row["representation"]),
            str(row["variant"]),
            str(row["split"]),
            int(row["horizon"]),
            int(row["context_length"]),
            str(row["signal"]),
        ),
    )


def _evaluate_gate(
    summary_rows: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    config: FeatureAblationConfig,
) -> dict[str, object]:
    primary_cells = [
        row
        for row in summary_rows
        if row["variant"] == config.primary_variant and row["split"] == config.primary_split
    ]
    robust_cells = sum(row["robust_improvement_cell"] is True for row in primary_cells)
    required_cells = math.ceil(
        config.minimum_robust_cell_rate * len(config.context_lengths) * len(config.horizons)
    )
    primary_seeds = [
        row
        for row in seed_rows
        if row["variant"] == config.primary_variant and row["split"] == config.primary_split
    ]
    capacity_seeds = [
        row
        for row in seed_rows
        if row["variant"] == config.capacity_control_variant
        and row["split"] == config.primary_split
    ]
    primary_successes = sum(row["primary_seed_success"] is True for row in primary_seeds)
    capacity_successes = sum(row["capacity_control_seed_success"] is True for row in capacity_seeds)
    required_seeds = math.ceil(config.robust_seed_rate * len(config.seeds))
    complete = (
        len(primary_cells) == len(config.context_lengths) * len(config.horizons)
        and all(
            int(row["n_successes"]) == len(config.seeds) and int(row["n_failures"]) == 0
            for row in primary_cells
        )
        and len(primary_seeds) == len(config.seeds)
        and all(
            int(row["n_cells"]) == len(config.context_lengths) * len(config.horizons)
            for row in primary_seeds
        )
        and len(capacity_seeds) == len(config.seeds)
        and all(
            int(row["n_cells"]) == len(config.context_lengths) * len(config.horizons)
            for row in capacity_seeds
        )
    )
    checks = {
        "complete_primary_grid": complete,
        "primary_seed_rate": primary_successes >= required_seeds,
        "robust_cell_coverage": robust_cells >= required_cells,
        "capacity_control_seed_rate": capacity_successes >= required_seeds,
    }
    test_cells = [
        row
        for row in summary_rows
        if row["variant"] == config.primary_variant and row["split"] == "test"
    ]
    return {
        "status": "evaluated",
        "passed": all(checks.values()),
        "primary_split": config.primary_split,
        "reference_variant": config.reference_variant,
        "primary_variant": config.primary_variant,
        "capacity_control_variant": config.capacity_control_variant,
        "criteria": {
            "maximum_primary_rmse_ratio": config.maximum_primary_rmse_ratio,
            "maximum_capacity_control_rmse_ratio": config.maximum_capacity_control_rmse_ratio,
            "robust_seed_rate": config.robust_seed_rate,
            "minimum_robust_cell_rate": config.minimum_robust_cell_rate,
            "required_seeds": required_seeds,
            "required_cells": required_cells,
        },
        "observed": {
            "primary_seed_successes": primary_successes,
            "primary_seed_trials": len(primary_seeds),
            "robust_primary_cells": robust_cells,
            "primary_cell_trials": len(primary_cells),
            "capacity_control_seed_successes": capacity_successes,
            "capacity_control_seed_trials": len(capacity_seeds),
            "test_robust_primary_cells_descriptive": sum(
                row["robust_improvement_cell"] is True for row in test_cells
            ),
            "test_primary_cell_trials_descriptive": len(test_cells),
        },
        "checks": checks,
    }


def _plot_summaries(
    condition_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    config: FeatureAblationConfig,
    plots_dir: Path,
    run_id: str,
) -> None:
    import matplotlib.pyplot as plt

    primary = [row for row in summary_rows if row["variant"] == config.primary_variant]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    image = None
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        selected = [row for row in primary if row["split"] == split]
        matrix = np.asarray(
            [
                [
                    float(
                        next(
                            row["rmse_ratio_vs_reference_median"]
                            for row in selected
                            if int(row["context_length"]) == context
                            and int(row["horizon"]) == horizon
                        )
                    )
                    for horizon in config.horizons
                ]
                for context in config.context_lengths
            ]
        )
        image = axis.imshow(matrix, vmin=0.95, vmax=1.05, cmap="coolwarm")
        axis.set_xticks(range(len(config.horizons)), config.horizons)
        axis.set_yticks(range(len(config.context_lengths)), config.context_lengths)
        axis.set_xlabel("Horizon")
        axis.set_ylabel("Context length")
        axis.set_title(split.title())
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                )
    if image is None:
        raise RuntimeError("primary heatmap requires completed summaries")
    figure.colorbar(image, ax=axes.tolist(), label="Median RMSE ratio vs absolute geometry")
    figure.suptitle(f"Incremental turning-feature effect\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__primary-effect-heatmap.png", dpi=config.plot_dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True, sharey=True)
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        distributions: list[list[float]] = []
        labels: list[str] = []
        for variant in config.variant_names:
            rows = [
                row
                for row in condition_rows
                if row["variant"] == variant and row["split"] == split and row["status"] == "ok"
            ]
            distributions.append([float(row["rmse_ratio_vs_reference"]) for row in rows])
            labels.append(variant.replace("_", "\n"))
        axis.boxplot(distributions, tick_labels=labels, showmeans=True)
        axis.axhline(1.0, color="black", linestyle="--")
        axis.axhline(config.maximum_primary_rmse_ratio, color="black", linestyle=":")
        axis.set_title(split.title())
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("RMSE ratio vs absolute geometry")
    figure.suptitle(f"Feature-ablation distributions\nrun={run_id}", fontsize=10.0)
    figure.savefig(plots_dir / "summary__variant-ratio-distributions.png", dpi=config.plot_dpi)
    plt.close(figure)

    test_vector = [
        row
        for row in summary_rows
        if row["representation"] == "vectorchain" and row["split"] == "test"
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for variant in config.variant_names:
        rows = [row for row in test_vector if row["variant"] == variant]
        axis.scatter(
            [float(row["mean_input_scalar_elements"]) for row in rows],
            [float(row["rmse_ratio_vs_raw_mean"]) for row in rows],
            alpha=0.75,
            label=variant,
        )
    axis.axhline(config.predictive_parity_ratio, color="black", linestyle="--")
    axis.set_xlabel("Mean float64 elements before pooling")
    axis.set_ylabel("Mean test RMSE ratio vs raw")
    axis.set_title(f"Payload-prediction tradeoff\nrun={run_id}", fontsize=10.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(plots_dir / "summary__payload-error-tradeoff.png", dpi=config.plot_dpi)
    plt.close(figure)

    selected_seed_rows = [
        row
        for row in seed_rows
        if row["split"] == config.primary_split
        and row["variant"] in {config.primary_variant, config.capacity_control_variant}
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for variant, marker in (
        (config.primary_variant, "o"),
        (config.capacity_control_variant, "s"),
    ):
        rows = sorted(
            (row for row in selected_seed_rows if row["variant"] == variant),
            key=lambda row: int(row["seed"]),
        )
        axis.plot(
            [str(row["seed"]) for row in rows],
            [float(row["geometric_mean_rmse_ratio_vs_reference"]) for row in rows],
            marker=marker,
            label=variant,
        )
    axis.axhline(config.maximum_primary_rmse_ratio, color="black", linestyle="--")
    axis.axhline(1.0, color="black", linestyle=":")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Geometric mean RMSE ratio vs reference")
    axis.set_title(f"Validation effect by independent seed\nrun={run_id}", fontsize=10.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__seed-geometric-ratios.png", dpi=config.plot_dpi)
    plt.close(figure)


def _write_partial(
    run_dir: Path,
    condition_rows: Sequence[Mapping[str, object]],
    signal_rows: Sequence[Mapping[str, object]],
    step_rows: Sequence[Mapping[str, object]],
) -> None:
    forecasting._write_csv(run_dir / "conditions.csv", CONDITION_FIELDS, condition_rows)
    forecasting._write_csv(
        run_dir / "conditions_by_signal.csv", SIGNAL_CONDITION_FIELDS, signal_rows
    )
    forecasting._write_csv(run_dir / "step_audit.csv", STEP_AUDIT_FIELDS, step_rows)


def _condition_id(
    representation: str,
    variant: str,
    seed: int,
    context_length: int,
    horizon: int,
) -> str:
    suffix = variant or representation
    return f"seed-{seed}__h-{horizon}__c-{context_length}__{suffix}"


def _condition_key(row: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        int(row["seed"]),
        int(row["horizon"]),
        int(row["context_length"]),
        str(row["split"]),
    )


def _signal_key(row: Mapping[str, object]) -> tuple[int, int, int, str, str]:
    return (*_condition_key(row), str(row["signal"]))


def _role(representation: str, variant: str, config: FeatureAblationConfig) -> str:
    if representation != "vectorchain":
        return "baseline"
    if variant == config.reference_variant:
        return "reference"
    if variant == config.primary_variant:
        return "primary"
    if variant == config.capacity_control_variant:
        return "capacity_control"
    return "secondary"


def _configured_variant(table: Mapping[str, Any], name: str, variants: Sequence[str]) -> str:
    value = forecasting._non_empty_string(table.get(name), f"ablations.{name}")
    if value not in variants:
        msg = f"ablations.{name} must name a configured variant"
        raise ValueError(msg)
    return value


def _unit_interval(value: object, name: str, *, exclude_zero: bool) -> float:
    result = forecasting._real(value, name, minimum=0.0)
    if result > 1.0 or (exclude_zero and result == 0.0):
        operator = "0 < value <= 1" if exclude_zero else "0 <= value <= 1"
        msg = f"{name} must satisfy {operator}"
        raise ValueError(msg)
    return result


def _values(rows: Sequence[Mapping[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _optional_values(rows: Sequence[Mapping[str, object]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row.get(field, "") != ""],
        dtype=np.float64,
    )


def _optional_mean(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    values = _optional_values(rows, field)
    return float(np.mean(values)) if values.size else ""


def _boolean_rate(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
    comparable = [bool(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(comparable)) if comparable else ""


def _geometric_mean(values: np.ndarray) -> float:
    if values.size == 0 or np.any(values <= 0.0):
        msg = "geometric mean requires positive values"
        raise ValueError(msg)
    return float(np.exp(np.mean(np.log(values))))


def _summary_sort_key(row: Mapping[str, object]) -> tuple[str, str, str, int, int]:
    return (
        str(row["representation"]),
        str(row["variant"]),
        str(row["split"]),
        int(row["horizon"]),
        int(row["context_length"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting/feature_ablation.toml"),
        help="TOML forecasting feature-ablation configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line forecasting feature ablation."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(
        sys.argv if argv is None else ("06_forecasting_feature_ablation.py", *argv)
    )
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
                "gate_passed": summary.gate_passed,
            },
            sort_keys=True,
        )
    )
    return 1 if summary.n_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
