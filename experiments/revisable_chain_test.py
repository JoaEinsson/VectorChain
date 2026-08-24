"""Lock-bound, one-way canonical test runner for the pre-registered K7 gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import revisable_chain
import revisable_chain_validation as validation

CANONICAL_SCOPE = "canonical_test"
EXPECTED_SELECTION_LOCK_SHA256 = "d4e3e4ba8b03e4e8e3ca638cc2061790f84e81475a44cdc5713fb331525f58b2"
REVISION_RATIO_MAX = 0.99
TEMPORAL_RATIO_MAX = 0.99
RAW_RATIO_MAX = 1.05
ENERGY_RATIO_MIN = 1.25
REQUIRED_SEED_PASSES = 4
REQUIRED_MECHANISM_PASSES = 2
REQUIRED_HORIZON_PASSES = 2
REQUIRED_HORIZON_SEED_PASSES = 4
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 7_817_596_609_602_243_496
BOOTSTRAP_LOWER_QUANTILE = 0.025
BOOTSTRAP_UPPER_QUANTILE = 0.975
PLOT_DPI = 150

COMPARISONS = (
    ("K7-R", "revisable_absolute", "immutable_absolute", REVISION_RATIO_MAX),
    ("K7-D-absolute", "revisable_temporal", "revisable_absolute", TEMPORAL_RATIO_MAX),
    ("K7-D-spatial", "revisable_temporal", "revisable_spatial", TEMPORAL_RATIO_MAX),
    ("K7-U", "revisable_temporal", "raw_matched", RAW_RATIO_MAX),
)


@dataclass(frozen=True, slots=True)
class TestConfig:
    """Validated canonical K7 test configuration and frozen selection."""

    name: str
    phase: str
    scope: str
    selection_config_path: Path
    selection_lock_path: Path
    selection_lock_sha256: str
    selection_config: validation.ValidationConfig
    selection: Mapping[str, Any]
    output_root: str
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TestRunSummary:
    """Identity and scientific outcome of one primary or replication run."""

    run_id: str
    run_dir: Path
    gate_path: Path
    gate_passed: bool
    mode: str
    n_failures: int


def load_config(path: Path) -> TestConfig:
    """Load and verify the immutable K7 test contract without generating signals."""

    resolved_path = path.resolve()
    with resolved_path.open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = _table(raw, "experiment")
    selection_table = _table(raw, "selection")
    gate = _table(raw, "gate")
    bootstrap = _table(raw, "bootstrap")
    output = _table(raw, "output")

    name = _non_empty_string(experiment.get("name"), "experiment.name")
    phase = _non_empty_string(experiment.get("phase"), "experiment.phase")
    scope = _non_empty_string(experiment.get("scope"), "experiment.scope")
    if scope != CANONICAL_SCOPE:
        msg = f"experiment.scope must equal {CANONICAL_SCOPE!r}"
        raise ValueError(msg)

    selection_config_path = _relative_file(
        resolved_path, selection_table.get("config"), "selection.config"
    )
    selection_lock_path = _relative_file(
        resolved_path, selection_table.get("lock"), "selection.lock"
    )
    declared_lock_hash = _sha256_string(selection_table.get("lock_sha256"), "selection.lock_sha256")
    if declared_lock_hash != EXPECTED_SELECTION_LOCK_SHA256:
        msg = "selection.lock_sha256 does not match the pre-test frozen lock"
        raise ValueError(msg)
    actual_lock_hash = hashlib.sha256(selection_lock_path.read_bytes()).hexdigest()
    if actual_lock_hash != declared_lock_hash:
        msg = "selection lock content does not match selection.lock_sha256"
        raise ValueError(msg)

    selection_config = validation.load_config(selection_config_path)
    if selection_config.scope != "canonical_selection":
        msg = "selection.config must be the canonical train/validation configuration"
        raise ValueError(msg)
    selection_config_hash = hashlib.sha256(selection_config_path.read_bytes()).hexdigest()
    selection_payload = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    _validate_selection_lock(selection_payload, selection_config_hash)

    _require_exact_real(gate, "revision_ratio_max", REVISION_RATIO_MAX)
    _require_exact_real(gate, "temporal_ratio_max", TEMPORAL_RATIO_MAX)
    _require_exact_real(gate, "raw_ratio_max", RAW_RATIO_MAX)
    _require_exact_real(gate, "energy_ratio_min", ENERGY_RATIO_MIN)
    _require_exact_integer(gate, "required_seed_passes", REQUIRED_SEED_PASSES)
    _require_exact_integer(gate, "required_mechanism_passes", REQUIRED_MECHANISM_PASSES)
    _require_exact_integer(gate, "required_horizon_passes", REQUIRED_HORIZON_PASSES)
    _require_exact_integer(gate, "required_horizon_seed_passes", REQUIRED_HORIZON_SEED_PASSES)
    _require_exact_integer(bootstrap, "repetitions", BOOTSTRAP_REPETITIONS)
    _require_exact_integer(bootstrap, "seed", BOOTSTRAP_SEED)
    _require_exact_real(bootstrap, "lower_quantile", BOOTSTRAP_LOWER_QUANTILE)
    _require_exact_real(bootstrap, "upper_quantile", BOOTSTRAP_UPPER_QUANTILE)

    output_root = _non_empty_string(output.get("root"), "output.root")
    save_plots = _boolean(output.get("save_plots"), "output.save_plots")
    if not save_plots:
        msg = "output.save_plots must remain true for the canonical artifact set"
        raise ValueError(msg)
    plot_dpi = _integer(output.get("plot_dpi"), "output.plot_dpi", minimum=1)
    if plot_dpi != PLOT_DPI:
        msg = f"output.plot_dpi must equal the registered value {PLOT_DPI}"
        raise ValueError(msg)

    return TestConfig(
        name=name,
        phase=phase,
        scope=scope,
        selection_config_path=selection_config_path,
        selection_lock_path=selection_lock_path,
        selection_lock_sha256=actual_lock_hash,
        selection_config=selection_config,
        selection=selection_payload,
        output_root=output_root,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )


def run_test(
    config_path: Path,
    *,
    output_root: Path | None = None,
    replication_of: Path | None = None,
    command_args: Sequence[str] = (),
) -> TestRunSummary:
    """Open the frozen test once, or explicitly reproduce its primary run."""

    config = load_config(config_path)
    repository_root = Path(__file__).resolve().parents[1]
    started = datetime.now(UTC)
    config_hash = hashlib.sha256(Path(config_path).resolve().read_bytes()).hexdigest()
    run_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash[:10]}"
    root = Path(output_root) if output_root is not None else repository_root / config.output_root
    experiment_root = root.resolve() / config.name
    mode = "replication" if replication_of is not None else "primary"
    primary_dir = Path(replication_of).resolve() if replication_of is not None else None
    _validate_run_authority(experiment_root, mode=mode, primary_dir=primary_dir)

    run_dir = experiment_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    train_end, test_start = _split_bounds(config.selection_config)
    git_commit, git_dirty = validation._git_state(repository_root)
    environment = _environment(
        config=config,
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        command_args=command_args,
        train_end=train_end,
        test_start=test_start,
        mode=mode,
        primary_dir=primary_dir,
    )
    validation._write_json(run_dir / "config.json", _resolved_config(config, train_end, test_start))
    validation._write_json(run_dir / "environment.json", environment)
    validation._write_csv(
        run_dir / "failures.csv", ("stage", "mechanism", "seed", "error_type", "message"), ()
    )
    run_started = time.perf_counter_ns()

    try:
        if git_dirty:
            msg = "canonical test and replication require a clean Git worktree"
            raise RuntimeError(msg)
        if primary_dir is not None:
            _validate_primary(primary_dir, git_commit, config)

        # This durable transition is written immediately before the first full signal exists.
        environment["test_opened"] = True
        environment["test_opened_utc"] = datetime.now(UTC).isoformat()
        validation._write_json(run_dir / "environment.json", environment)

        selected = _selected_penalties(config.selection)
        models, rows = _fit_and_test(
            config.selection_config,
            selected=selected,
            train_end=train_end,
            test_start=test_start,
        )
        validation._persist_validation_artifacts(run_dir, models=models, rows=rows)
        _persist_batch_stream_artifact(run_dir, rows)
        (run_dir / "selection.json").write_bytes(config.selection_lock_path.read_bytes())

        gate, gate_rows = evaluate_gate(
            rows["metrics"],
            rows["working_state"],
            rows["solver_audit"],
            rows["causality_audit"],
            rows["batch_stream_audit"],
            rows["commit_audit"],
            test_start=test_start,
        )
        _persist_gate_artifacts(run_dir, gate, gate_rows)
        if config.save_plots:
            _render_plots(run_dir, gate_rows, rows["working_state"], dpi=config.plot_dpi)
        if primary_dir is not None:
            reproduction = compare_scientific_runs(primary_dir, run_dir)
            validation._write_json(run_dir / "reproduction.json", reproduction)
            if not reproduction["scientifically_identical"]:
                msg = "replication differs from the primary run in scientific outputs"
                raise RuntimeError(msg)
    except Exception as error:
        validation._write_csv(
            run_dir / "failures.csv",
            ("stage", "mechanism", "seed", "error_type", "message"),
            (
                {
                    "stage": "canonical_test_runner",
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
        validation._write_json(run_dir / "environment.json", environment)
        validation._write_manifest(run_dir, status="failed")
        raise

    environment["status"] = "complete"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_started) / 1e9
    environment["n_series"] = len(config.selection_config.seeds) * len(
        config.selection_config.mechanisms
    )
    environment["n_failures"] = 0
    environment["test_materialized"] = True
    environment["gate_passed"] = bool(gate["passed"])
    validation._write_json(run_dir / "environment.json", environment)
    validation._write_manifest(run_dir, status="complete")
    return TestRunSummary(
        run_id=run_id,
        run_dir=run_dir,
        gate_path=run_dir / "gate.json",
        gate_passed=bool(gate["passed"]),
        mode=mode,
        n_failures=0,
    )


def evaluate_gate(
    metrics: Sequence[Mapping[str, object]],
    working_state: Sequence[Mapping[str, object]],
    solver_audit: Sequence[Mapping[str, object]],
    causality_audit: Sequence[Mapping[str, object]],
    batch_stream_audit: Sequence[Mapping[str, object]],
    commit_audit: Sequence[Mapping[str, object]],
    *,
    test_start: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, object]]]]:
    """Apply the exact pre-registered K7 aggregation and discrete criteria."""

    metric_lookup = {
        (
            str(row["mechanism"]),
            int(row["seed"]),
            int(row["horizon"]),
            str(row["representation"]),
        ): float(row["rmse"])
        for row in metrics
    }
    mechanisms = tuple(sorted({key[0] for key in metric_lookup}))
    seeds = tuple(sorted({key[1] for key in metric_lookup}))
    horizons = tuple(sorted({key[2] for key in metric_lookup}))
    _validate_canonical_dimensions(mechanisms, seeds, horizons)

    cells: list[dict[str, object]] = []
    by_seed: list[dict[str, object]] = []
    by_mechanism: list[dict[str, object]] = []
    by_horizon_seed: list[dict[str, object]] = []
    by_horizon: list[dict[str, object]] = []
    comparison_gates: dict[str, dict[str, Any]] = {}
    for comparison, candidate, control, threshold in COMPARISONS:
        for mechanism in mechanisms:
            for seed in seeds:
                for horizon in horizons:
                    candidate_rmse = metric_lookup[(mechanism, seed, horizon, candidate)]
                    control_rmse = metric_lookup[(mechanism, seed, horizon, control)]
                    ratio = _safe_ratio(candidate_rmse, control_rmse)
                    cells.append(
                        {
                            "comparison": comparison,
                            "candidate": candidate,
                            "control": control,
                            "mechanism": mechanism,
                            "seed": seed,
                            "horizon": horizon,
                            "candidate_rmse": candidate_rmse,
                            "control_rmse": control_rmse,
                            "ratio": ratio,
                            "threshold": threshold,
                            "passed": int(ratio <= threshold),
                        }
                    )
        summary = _aggregate_comparison(
            cells,
            comparison=comparison,
            threshold=threshold,
            mechanisms=mechanisms,
            seeds=seeds,
            horizons=horizons,
            by_seed=by_seed,
            by_mechanism=by_mechanism,
            by_horizon_seed=by_horizon_seed,
            by_horizon=by_horizon,
        )
        comparison_gates[comparison] = summary

    energy_cells, energy_by_seed, energy_by_mechanism, energy_gate = _energy_gate(
        working_state,
        mechanisms=mechanisms,
        seeds=seeds,
        test_start=test_start,
    )
    structural = _structural_gate(
        working_state,
        solver_audit,
        causality_audit,
        batch_stream_audit,
        commit_audit,
        mechanisms=mechanisms,
        seeds=seeds,
    )
    bootstrap = _bootstrap_comparisons(cells, mechanisms, seeds, horizons)
    k7_r_passed = bool(comparison_gates["K7-R"]["passed"])
    k7_d_passed = bool(
        comparison_gates["K7-D-absolute"]["passed"]
        and comparison_gates["K7-D-spatial"]["passed"]
        and energy_gate["passed"]
    )
    k7_u_passed = bool(comparison_gates["K7-U"]["passed"])
    passed = bool(k7_r_passed and k7_d_passed and k7_u_passed and structural["passed"])
    gate_payload: dict[str, Any] = {
        "status": "complete",
        "decisory_split": "test",
        "passed": passed,
        "aggregation": {
            "cell": "paired RMSE ratio per mechanism, seed and horizon",
            "operator": "geometric mean in log space",
            "seed": "three mechanisms times three horizons",
            "mechanism": "five seeds times three horizons",
            "horizon": "geometric mean across mechanisms per seed; robust in 4/5 seeds",
            "replicate_unit": "complete mechanism times seed series",
        },
        "subgates": {
            "K7-R": {"passed": k7_r_passed, "comparison": comparison_gates["K7-R"]},
            "K7-D": {
                "passed": k7_d_passed,
                "against_absolute": comparison_gates["K7-D-absolute"],
                "against_spatial": comparison_gates["K7-D-spatial"],
                "correction_energy": energy_gate,
            },
            "K7-U": {"passed": k7_u_passed, "comparison": comparison_gates["K7-U"]},
            "structural": structural,
        },
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "unit": "15 complete mechanism times seed series",
            "decisory": False,
        },
    }
    return gate_payload, {
        "comparison_cells": cells,
        "gate_by_seed": by_seed,
        "gate_by_mechanism": by_mechanism,
        "gate_by_horizon_seed": by_horizon_seed,
        "gate_by_horizon": by_horizon,
        "energy_cells": energy_cells,
        "energy_by_seed": energy_by_seed,
        "energy_by_mechanism": energy_by_mechanism,
        "bootstrap_summary": bootstrap,
        "structural_summary": [structural],
    }


def _fit_and_test(
    config: validation.ValidationConfig,
    *,
    selected: tuple[float, float],
    train_end: int,
    test_start: int,
) -> tuple[dict[str, validation.MultiRidgeModel], dict[str, list[dict[str, object]]]]:
    models: dict[str, validation.MultiRidgeModel] = {}
    rows: dict[str, list[dict[str, object]]] = {
        "origins": [],
        "inputs": [],
        "predictions": [],
        "metrics": [],
        "working_state": [],
        "working_joints": [],
        "events": [],
        "commit_audit": [],
        "causality_audit": [],
        "solver_audit": [],
        "batch_stream_audit": [],
    }
    for mechanism in config.mechanisms:
        for seed in config.seeds:
            signal = revisable_chain.generate_k7_signal(
                mechanism,
                seed=seed,
                n_points=config.n_points,
                noise_std=config.noise_std,
            )
            bundle = revisable_chain.build_k7_designs(
                signal,
                lambda_revision=selected[0],
                lambda_bend=selected[1],
            )
            train_mask = validation.endpoint_mask(
                bundle.target_indices, start_inclusive=0, end_exclusive=train_end
            )
            test_mask = validation.endpoint_mask(
                bundle.target_indices, start_inclusive=test_start, end_exclusive=config.n_points
            )
            validation._require_examples(train_mask, test_mask, mechanism, seed)
            _append_origin_rows(rows["origins"], bundle, train_mask, test_mask)
            validation._append_structural_rows(rows, bundle, mechanism, seed, train_end)
            _augment_working_state_raw_values(rows["working_state"], signal.values, mechanism, seed)
            _append_version_history(rows["working_joints"], rows["events"], bundle)
            _append_batch_stream_audit(rows["batch_stream_audit"], bundle, selected)

            for design in bundle.representations:
                _append_input_rows(rows["inputs"], design, bundle, train_mask, test_mask)
                if design.name == "persistence":
                    predictions = np.zeros(
                        (int(np.count_nonzero(test_mask)), len(revisable_chain.HORIZONS)),
                        dtype=np.float64,
                    )
                    input_rank = 0
                    scaler_state_bytes = 0
                    model_state_bytes = 0
                    train_runtime_s = 0.0
                else:
                    train_started = time.perf_counter_ns()
                    model = validation.fit_multi_ridge(
                        design.inputs[train_mask], bundle.targets[train_mask], alpha=config.alpha
                    )
                    train_runtime_s = (time.perf_counter_ns() - train_started) / 1e9
                    if model.n_predictive_parameters != revisable_chain.N_PREDICTIVE_PARAMETERS:
                        msg = "trained representation violated registered parameter parity"
                        raise RuntimeError(msg)
                    key = f"{mechanism}__seed_{seed}__{design.name}"
                    models[key] = model
                    predictions = model.predict(design.inputs[test_mask])
                    input_rank = int(np.linalg.matrix_rank(design.inputs[train_mask]))
                    scaler_state_bytes = model.scaler_state_bytes
                    model_state_bytes = model.state_bytes
                _append_prediction_metric_rows(
                    rows["predictions"],
                    rows["metrics"],
                    bundle,
                    design.name,
                    test_mask,
                    predictions,
                    input_rank=input_rank,
                    scaler_state_bytes=scaler_state_bytes,
                    model_state_bytes=model_state_bytes,
                    train_runtime_s=train_runtime_s,
                )
            validation._append_causality_row(
                rows["causality_audit"], bundle, mechanism, seed, selected
            )
    return models, rows


def _append_origin_rows(
    rows: list[dict[str, object]],
    bundle: revisable_chain.K7DesignBundle,
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
) -> None:
    for row_index, origin in enumerate(bundle.origins):
        rows.append(
            {
                "mechanism": bundle.signal.mechanism,
                "seed": bundle.signal.seed,
                "row_index": row_index,
                "origin": int(origin),
                "split": _row_split(row_index, train_mask, test_mask),
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
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
) -> None:
    for row_index, origin in enumerate(bundle.origins):
        values = design.inputs[row_index] if design.inputs.shape[1] else np.asarray(())
        row: dict[str, object] = {
            "mechanism": bundle.signal.mechanism,
            "seed": bundle.signal.seed,
            "representation": design.name,
            "row_index": row_index,
            "origin": int(origin),
            "split": _row_split(row_index, train_mask, test_mask),
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
    test_mask: NDArray[np.bool_],
    predictions: NDArray[np.float64],
    *,
    input_rank: int,
    scaler_state_bytes: int,
    model_state_bytes: int,
    train_runtime_s: float,
) -> None:
    selected_rows = np.flatnonzero(test_mask)
    actual = bundle.targets[test_mask]
    for local_index, row_index in enumerate(selected_rows):
        for column, horizon in enumerate(revisable_chain.HORIZONS):
            prediction_rows.append(
                {
                    "mechanism": bundle.signal.mechanism,
                    "seed": bundle.signal.seed,
                    "representation": representation,
                    "split": "test",
                    "origin": int(bundle.origins[row_index]),
                    "horizon": horizon,
                    "target_index": int(bundle.target_indices[row_index, column]),
                    "actual": float(actual[local_index, column]),
                    "predicted": float(predictions[local_index, column]),
                }
            )
    for column, horizon in enumerate(revisable_chain.HORIZONS):
        residual = predictions[:, column] - actual[:, column]
        trained = representation != "persistence"
        metric_rows.append(
            {
                "mechanism": bundle.signal.mechanism,
                "seed": bundle.signal.seed,
                "representation": representation,
                "split": "test",
                "horizon": horizon,
                "n_origins": predictions.shape[0],
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "input_steps": 0
                if not trained
                else revisable_chain.RAW_MATCHED_STEPS
                if representation == "raw_matched"
                else revisable_chain.VECTOR_STEPS,
                "input_scalars": revisable_chain.N_INPUT_SCALARS if trained else 0,
                "input_bytes": (
                    revisable_chain.N_INPUT_SCALARS * np.dtype(np.float64).itemsize
                    if trained
                    else 0
                ),
                "input_rank": input_rank,
                "n_predictive_parameters": (
                    revisable_chain.N_PREDICTIVE_PARAMETERS if trained else 0
                ),
                "scaler_state_bytes": scaler_state_bytes,
                "model_state_bytes": model_state_bytes,
                "train_runtime_s": train_runtime_s,
            }
        )


def _append_batch_stream_audit(
    rows: list[dict[str, object]],
    bundle: revisable_chain.K7DesignBundle,
    selected: tuple[float, float],
) -> None:
    batch = revisable_chain.RevisableVectorChain(
        tolerance=revisable_chain.TOLERANCE,
        min_segment_length=revisable_chain.MIN_SEGMENT_LENGTH,
        lambda_revision=selected[0],
        lambda_bend=selected[1],
    )
    batch_versions = batch.fit_transform(bundle.signal.values)
    versions_equal = batch_versions == bundle.versions
    committed_equal = batch.committed_ == bundle.committed
    events_equal = batch.events_ == bundle.events
    trained_designs = [design for design in bundle.representations if design.name != "persistence"]
    aligned_rows = all(design.inputs.shape[0] == bundle.origins.size for design in trained_designs)
    finite_inputs = all(np.all(np.isfinite(design.inputs)) for design in trained_designs)
    payload_parity = all(
        design.scalar_elements == revisable_chain.N_INPUT_SCALARS
        and design.predictive_parameters == revisable_chain.N_PREDICTIVE_PARAMETERS
        and design.input_steps
        == (
            revisable_chain.RAW_MATCHED_STEPS
            if design.name == "raw_matched"
            else revisable_chain.VECTOR_STEPS
        )
        for design in trained_designs
    )
    passed = bool(
        versions_equal
        and committed_equal
        and events_equal
        and aligned_rows
        and finite_inputs
        and payload_parity
    )
    if not passed:
        msg = (
            f"batch/stream or representation audit failed for {bundle.signal.mechanism}/"
            f"seed={bundle.signal.seed}"
        )
        raise RuntimeError(msg)
    rows.append(
        {
            "mechanism": bundle.signal.mechanism,
            "seed": bundle.signal.seed,
            "n_versions": len(batch_versions),
            "versions_equal": int(versions_equal),
            "committed_equal": int(committed_equal),
            "events_equal": int(events_equal),
            "representation_rows_aligned": int(aligned_rows),
            "all_inputs_finite": int(finite_inputs),
            "payload_parity": int(payload_parity),
            "shared_origins_targets_boundaries": 1,
            "passed": int(passed),
        }
    )


def _append_version_history(
    joint_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    bundle: revisable_chain.K7DesignBundle,
) -> None:
    for item in bundle.versions:
        for joint in item.joints:
            joint_rows.append(
                {
                    "mechanism": bundle.signal.mechanism,
                    "seed": bundle.signal.seed,
                    "observed_at": item.observed_at,
                    "version": item.version,
                    "joint_id": joint.joint_id,
                    "sample_index": joint.sample_index,
                    "value": joint.value,
                    "raw_value": float(bundle.signal.values[joint.sample_index]),
                    "created_at": joint.created_at,
                }
            )
    for event in bundle.events:
        event_rows.append(
            {
                "mechanism": bundle.signal.mechanism,
                "seed": bundle.signal.seed,
                "kind": event.kind,
                "observed_at": event.observed_at,
                "version": event.version,
                "target_id": "" if event.target_id is None else event.target_id,
            }
        )


def _augment_working_state_raw_values(
    rows: list[dict[str, object]],
    values: NDArray[np.float64],
    mechanism: str,
    seed: int,
) -> None:
    for row in reversed(rows):
        if row["mechanism"] != mechanism or int(row["seed"]) != seed:
            break
        row["raw_start_value"] = float(values[int(row["start"])])
        row["raw_end_value"] = float(values[int(row["end"])])


def _aggregate_comparison(
    cells: Sequence[Mapping[str, object]],
    *,
    comparison: str,
    threshold: float,
    mechanisms: Sequence[str],
    seeds: Sequence[int],
    horizons: Sequence[int],
    by_seed: list[dict[str, object]],
    by_mechanism: list[dict[str, object]],
    by_horizon_seed: list[dict[str, object]],
    by_horizon: list[dict[str, object]],
) -> dict[str, Any]:
    selected = [row for row in cells if row["comparison"] == comparison]
    seed_passes = 0
    for seed in seeds:
        ratio = _geometric_mean(float(row["ratio"]) for row in selected if row["seed"] == seed)
        passed = ratio <= threshold
        seed_passes += int(passed)
        by_seed.append(
            {
                "comparison": comparison,
                "seed": seed,
                "geometric_mean_ratio": ratio,
                "threshold": threshold,
                "passed": int(passed),
            }
        )
    mechanism_passes = 0
    for mechanism in mechanisms:
        ratio = _geometric_mean(
            float(row["ratio"]) for row in selected if row["mechanism"] == mechanism
        )
        passed = ratio <= threshold
        mechanism_passes += int(passed)
        by_mechanism.append(
            {
                "comparison": comparison,
                "mechanism": mechanism,
                "geometric_mean_ratio": ratio,
                "threshold": threshold,
                "passed": int(passed),
            }
        )
    robust_horizons = 0
    horizon_seed_counts: dict[int, int] = {}
    for horizon in horizons:
        horizon_seed_passes = 0
        for seed in seeds:
            ratio = _geometric_mean(
                float(row["ratio"])
                for row in selected
                if row["horizon"] == horizon and row["seed"] == seed
            )
            passed = ratio <= threshold
            horizon_seed_passes += int(passed)
            by_horizon_seed.append(
                {
                    "comparison": comparison,
                    "horizon": horizon,
                    "seed": seed,
                    "geometric_mean_ratio": ratio,
                    "threshold": threshold,
                    "passed": int(passed),
                }
            )
        robust = horizon_seed_passes >= REQUIRED_HORIZON_SEED_PASSES
        robust_horizons += int(robust)
        horizon_seed_counts[horizon] = horizon_seed_passes
        by_horizon.append(
            {
                "comparison": comparison,
                "horizon": horizon,
                "seed_passes": horizon_seed_passes,
                "required_seed_passes": REQUIRED_HORIZON_SEED_PASSES,
                "passed": int(robust),
            }
        )
    passed = bool(
        seed_passes >= REQUIRED_SEED_PASSES
        and mechanism_passes >= REQUIRED_MECHANISM_PASSES
        and robust_horizons >= REQUIRED_HORIZON_PASSES
    )
    return {
        "passed": passed,
        "threshold_max": threshold,
        "seed_passes": seed_passes,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "mechanism_passes": mechanism_passes,
        "required_mechanism_passes": REQUIRED_MECHANISM_PASSES,
        "robust_horizons": robust_horizons,
        "required_robust_horizons": REQUIRED_HORIZON_PASSES,
        "horizon_seed_passes": {str(key): value for key, value in horizon_seed_counts.items()},
    }


def _energy_gate(
    working_state: Sequence[Mapping[str, object]],
    *,
    mechanisms: Sequence[str],
    seeds: Sequence[int],
    test_start: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, Any],
]:
    cells: list[dict[str, object]] = []
    for mechanism in mechanisms:
        for seed in seeds:
            series = [
                row
                for row in working_state
                if row["mechanism"] == mechanism
                and row["seed"] == seed
                and int(row["observed_at"]) >= test_start
            ]
            changing = [
                float(row["correction_energy"])
                for row in series
                if row["latent_region"] == "changing"
            ]
            stationary = [
                float(row["correction_energy"])
                for row in series
                if row["latent_region"] == "stationary"
            ]
            if not changing or not stationary:
                msg = f"missing changing/stationary energy values for {mechanism}/seed={seed}"
                raise RuntimeError(msg)
            changing_mean = float(np.mean(changing))
            stationary_mean = float(np.mean(stationary))
            ratio = _safe_ratio(changing_mean, stationary_mean)
            cells.append(
                {
                    "mechanism": mechanism,
                    "seed": seed,
                    "n_changing_links": len(changing),
                    "n_stationary_links": len(stationary),
                    "changing_mean_energy": changing_mean,
                    "stationary_mean_energy": stationary_mean,
                    "ratio": ratio,
                    "threshold": ENERGY_RATIO_MIN,
                    "passed": int(ratio >= ENERGY_RATIO_MIN),
                }
            )
    by_seed: list[dict[str, object]] = []
    seed_passes = 0
    for seed in seeds:
        ratio = _geometric_mean(float(row["ratio"]) for row in cells if row["seed"] == seed)
        passed = ratio >= ENERGY_RATIO_MIN
        seed_passes += int(passed)
        by_seed.append(
            {
                "seed": seed,
                "geometric_mean_ratio": ratio,
                "threshold": ENERGY_RATIO_MIN,
                "passed": int(passed),
            }
        )
    by_mechanism: list[dict[str, object]] = []
    mechanism_passes = 0
    for mechanism in mechanisms:
        ratio = _geometric_mean(
            float(row["ratio"]) for row in cells if row["mechanism"] == mechanism
        )
        passed = ratio >= ENERGY_RATIO_MIN
        mechanism_passes += int(passed)
        by_mechanism.append(
            {
                "mechanism": mechanism,
                "geometric_mean_ratio": ratio,
                "threshold": ENERGY_RATIO_MIN,
                "passed": int(passed),
            }
        )
    gate = {
        "passed": bool(
            seed_passes >= REQUIRED_SEED_PASSES and mechanism_passes >= REQUIRED_MECHANISM_PASSES
        ),
        "threshold_min": ENERGY_RATIO_MIN,
        "seed_passes": seed_passes,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "mechanism_passes": mechanism_passes,
        "required_mechanism_passes": REQUIRED_MECHANISM_PASSES,
        "aggregation": "arithmetic mean within region, geometric mean across series",
        "zero_corrections_included": True,
    }
    return cells, by_seed, by_mechanism, gate


def _structural_gate(
    working_state: Sequence[Mapping[str, object]],
    solver_audit: Sequence[Mapping[str, object]],
    causality_audit: Sequence[Mapping[str, object]],
    batch_stream_audit: Sequence[Mapping[str, object]],
    commit_audit: Sequence[Mapping[str, object]],
    *,
    mechanisms: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    expected_series = len(mechanisms) * len(seeds)
    finite_features = all(
        np.isfinite(float(row[key]))
        for row in working_state
        for key in (
            "start_value",
            "end_value",
            "dy",
            "theta",
            "r",
            "update_theta",
            "update_r",
            "correction_energy",
        )
    )
    positive_integer_dt = all(
        isinstance(row["dt"], int) and int(row["dt"]) > 0 for row in working_state
    )
    solver_pass = bool(solver_audit) and all(
        int(row["structural_pass"]) == 1 for row in solver_audit
    )
    causality_pass = len(causality_audit) == expected_series and all(
        int(row["passed"]) == 1 for row in causality_audit
    )
    batch_stream_pass = len(batch_stream_audit) == expected_series and all(
        int(row["passed"]) == 1 for row in batch_stream_audit
    )
    committed_immutable = bool(commit_audit) and all(
        int(row["immutable_snapshot"]) == 1 for row in commit_audit
    )
    max_links = max(int(row["n_links"]) for row in solver_audit)
    max_raw_span = max(int(row["raw_span"]) for row in solver_audit)
    max_start_error = max(float(row["start_anchor_error"]) for row in solver_audit)
    max_current_error = max(float(row["current_anchor_error"]) for row in solver_audit)
    passed = bool(
        finite_features
        and positive_integer_dt
        and solver_pass
        and causality_pass
        and batch_stream_pass
        and committed_immutable
        and max_links <= 4
        and max_raw_span <= 256
        and max_start_error <= 1e-12
        and max_current_error <= 1e-12
    )
    return {
        "passed": passed,
        "finite_features_and_solutions": finite_features,
        "positive_integer_dt": positive_integer_dt,
        "solver_and_bounds_passed": solver_pass,
        "causality_suffix_audits_passed": causality_pass,
        "batch_stream_and_determinism_passed": batch_stream_pass,
        "committed_snapshots_immutable": committed_immutable,
        "n_series": expected_series,
        "n_working_link_states": len(working_state),
        "n_solver_states": len(solver_audit),
        "n_committed_links": len(commit_audit),
        "max_working_links": max_links,
        "max_raw_span": max_raw_span,
        "max_start_anchor_error": max_start_error,
        "max_current_anchor_error": max_current_error,
    }


def _bootstrap_comparisons(
    cells: Sequence[Mapping[str, object]],
    mechanisms: Sequence[str],
    seeds: Sequence[int],
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for comparison, _candidate, _control, _threshold in COMPARISONS:
        selected = [row for row in cells if row["comparison"] == comparison]
        series_ratios = np.asarray(
            [
                _geometric_mean(
                    float(row["ratio"])
                    for row in selected
                    if row["mechanism"] == mechanism and row["seed"] == seed
                )
                for mechanism in mechanisms
                for seed in seeds
            ],
            dtype=np.float64,
        )
        if series_ratios.size != len(mechanisms) * len(seeds):
            msg = "bootstrap did not receive all 15 complete series"
            raise RuntimeError(msg)
        indices = rng.integers(
            0,
            series_ratios.size,
            size=(BOOTSTRAP_REPETITIONS, series_ratios.size),
        )
        samples = np.exp(np.mean(np.log(series_ratios[indices]), axis=1))
        rows.append(
            {
                "comparison": comparison,
                "n_series": series_ratios.size,
                "n_horizons_per_series": len(horizons),
                "point_geometric_mean_ratio": _geometric_mean(series_ratios),
                "lower": float(np.quantile(samples, BOOTSTRAP_LOWER_QUANTILE)),
                "median": float(np.quantile(samples, 0.5)),
                "upper": float(np.quantile(samples, BOOTSTRAP_UPPER_QUANTILE)),
                "lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
                "upper_quantile": BOOTSTRAP_UPPER_QUANTILE,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
                "decisory": 0,
            }
        )
    return rows


def _persist_gate_artifacts(
    run_dir: Path,
    gate: Mapping[str, Any],
    rows: Mapping[str, list[dict[str, object]]],
) -> None:
    validation._write_json(run_dir / "gate.json", gate)
    structural = rows["structural_summary"][0]
    validation._write_json(run_dir / "structural_summary.json", structural)
    for name, values in rows.items():
        if name == "structural_summary":
            continue
        validation._write_csv_from_rows(run_dir / f"{name}.csv", values)


def compare_scientific_runs(primary_dir: Path, replica_dir: Path) -> dict[str, Any]:
    """Compare deterministic scientific content while excluding runtime fields."""

    primary = Path(primary_dir).resolve()
    replica = Path(replica_dir).resolve()
    exact_files = (
        "config.json",
        "selection.json",
        "origins.csv",
        "predictions.csv",
        "solver_audit.csv",
        "commit_audit.csv",
        "causality_audit.csv",
        "batch_stream_audit.csv",
        "comparison_cells.csv",
        "gate_by_seed.csv",
        "gate_by_mechanism.csv",
        "gate_by_horizon_seed.csv",
        "gate_by_horizon.csv",
        "energy_cells.csv",
        "energy_by_seed.csv",
        "energy_by_mechanism.csv",
        "bootstrap_summary.csv",
        "gate.json",
        "structural_summary.json",
    )
    comparisons: list[dict[str, object]] = []
    for name in exact_files:
        same = (primary / name).read_bytes() == (replica / name).read_bytes()
        comparisons.append({"artifact": name, "comparison": "exact_bytes", "passed": same})
    for name in (
        "inputs.csv.gz",
        "working_state.csv.gz",
        "working_joints.csv.gz",
        "events.csv.gz",
    ):
        with gzip.open(primary / name, "rb") as left, gzip.open(replica / name, "rb") as right:
            same = left.read() == right.read()
        comparisons.append(
            {"artifact": name, "comparison": "decompressed_exact_bytes", "passed": same}
        )
    primary_metrics = _csv_without_columns(primary / "metrics.csv", {"train_runtime_s"})
    replica_metrics = _csv_without_columns(replica / "metrics.csv", {"train_runtime_s"})
    comparisons.append(
        {
            "artifact": "metrics.csv",
            "comparison": "rows excluding train_runtime_s",
            "passed": primary_metrics == replica_metrics,
        }
    )
    with np.load(primary / "models.npz") as left, np.load(replica / "models.npz") as right:
        same_models = left.files == right.files and all(
            np.array_equal(left[name], right[name]) for name in left.files
        )
    comparisons.append(
        {"artifact": "models.npz", "comparison": "exact arrays", "passed": same_models}
    )
    return {
        "status": "complete",
        "scientifically_identical": all(bool(row["passed"]) for row in comparisons),
        "primary_run": str(primary),
        "replica_run": str(replica),
        "excluded": [
            "environment timestamps and elapsed runtime",
            "metrics.csv train_runtime_s",
            "manifest byte hashes affected by excluded runtime fields",
            "plot encodings; plots are derived from compared tables",
        ],
        "comparisons": comparisons,
    }


def _render_plots(
    run_dir: Path,
    gate_rows: Mapping[str, list[dict[str, object]]],
    working_state: Sequence[Mapping[str, object]],
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    cells = gate_rows["comparison_cells"]
    labels = [item[0] for item in COMPARISONS]
    data = [
        [float(row["ratio"]) for row in cells if row["comparison"] == label] for label in labels
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(data, tick_labels=labels, showfliers=False)
    for index, (_label, _candidate, _control, threshold) in enumerate(COMPARISONS, start=1):
        axis.plot((index - 0.35, index + 0.35), (threshold, threshold), "r--", linewidth=1)
    axis.set_ylabel("paired RMSE ratio")
    axis.set_title("K7 test ratios by pre-registered comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "paired_error_ratios.png", dpi=dpi)
    plt.close(figure)

    energy = gate_rows["energy_cells"]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(energy))
    axis.scatter(x, [float(row["ratio"]) for row in energy], s=24)
    axis.axhline(ENERGY_RATIO_MIN, color="red", linestyle="--", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{str(row['mechanism']).split('_')[0]}\n{row['seed']}" for row in energy],
        rotation=90,
        fontsize=7,
    )
    axis.set_ylabel("changing / stationary correction energy")
    axis.set_title("Correction-energy localization in the closed test")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "correction_energy.png", dpi=dpi)
    plt.close(figure)

    first_mechanism = str(working_state[0]["mechanism"])
    first_seed = int(working_state[0]["seed"])
    observed_at = max(
        int(row["observed_at"])
        for row in working_state
        if row["mechanism"] == first_mechanism and row["seed"] == first_seed
    )
    chain = [
        row
        for row in working_state
        if row["mechanism"] == first_mechanism
        and int(row["seed"]) == first_seed
        and int(row["observed_at"]) == observed_at
    ]
    immutable_x = [int(chain[0]["start"]), *(int(row["end"]) for row in chain)]
    immutable_y = [
        float(chain[0]["raw_start_value"]),
        *(float(row["raw_end_value"]) for row in chain),
    ]
    revised_x = [int(chain[0]["start"]), *(int(row["end"]) for row in chain)]
    revised_y = [float(chain[0]["start_value"]), *(float(row["end_value"]) for row in chain)]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(immutable_x, immutable_y, "o--", label="immutable joints")
    axis.plot(revised_x, revised_y, "o-", label="revised joints")
    axis.set_xlabel("raw sample index")
    axis.set_ylabel("joint value")
    axis.set_title(f"Working chain before/after revision: {first_mechanism}, seed {first_seed}")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "chain_before_after.png", dpi=dpi)
    plt.close(figure)


def _persist_batch_stream_artifact(
    run_dir: Path, rows: Mapping[str, list[dict[str, object]]]
) -> None:
    validation._write_csv_from_rows(run_dir / "batch_stream_audit.csv", rows["batch_stream_audit"])
    validation._write_gzip_csv_from_rows(run_dir / "working_joints.csv.gz", rows["working_joints"])
    validation._write_gzip_csv_from_rows(run_dir / "events.csv.gz", rows["events"])


def _validate_run_authority(experiment_root: Path, *, mode: str, primary_dir: Path | None) -> None:
    if mode == "replication":
        if primary_dir is None or not primary_dir.is_dir():
            msg = "--replicate must reference an existing primary run directory"
            raise ValueError(msg)
        return
    if not experiment_root.exists():
        return
    for environment_path in experiment_root.glob("*/environment.json"):
        try:
            payload = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("mode") == "primary" and payload.get("test_opened") is True:
            msg = (
                "the canonical test was already opened; only --replicate of that primary is allowed"
            )
            raise RuntimeError(msg)


def _validate_primary(primary_dir: Path, git_commit: str, config: TestConfig) -> None:
    environment_path = primary_dir / "environment.json"
    if not environment_path.is_file():
        msg = "replication primary is missing environment.json"
        raise ValueError(msg)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment.get("mode") != "primary" or environment.get("status") != "complete":
        msg = "replication requires a completed primary test run"
        raise ValueError(msg)
    if environment.get("git", {}).get("commit") != git_commit:
        msg = "replication must run on the exact primary Git commit"
        raise RuntimeError(msg)
    if environment.get("selection_lock_sha256") != config.selection_lock_sha256:
        msg = "replication selection lock differs from the primary run"
        raise RuntimeError(msg)
    if (primary_dir / "config.json").read_bytes() != json.dumps(
        _resolved_config(config, *_split_bounds(config.selection_config)),
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n":
        msg = "replication configuration differs from the primary run"
        raise RuntimeError(msg)


def _split_bounds(config: validation.ValidationConfig) -> tuple[int, int]:
    train_end = int(config.n_points * config.train_fraction)
    test_start = int(config.n_points * (config.train_fraction + config.validation_fraction))
    if train_end <= 0 or test_start <= train_end or test_start >= config.n_points:
        msg = "selection split does not define the registered closed test suffix"
        raise ValueError(msg)
    return train_end, test_start


def _resolved_config(config: TestConfig, train_end: int, test_start: int) -> dict[str, object]:
    selected = _selected_penalties(config.selection)
    return {
        "source": config.raw,
        "resolved": {
            "scope": config.scope,
            "selection_config": str(config.selection_config_path),
            "selection_config_sha256": config.selection["source"]["config_sha256"],
            "selection_lock": str(config.selection_lock_path),
            "selection_lock_sha256": config.selection_lock_sha256,
            "lambda_revision": selected[0],
            "lambda_bend": selected[1],
            "train_end_exclusive": train_end,
            "validation_start_inclusive": train_end,
            "test_start_inclusive": test_start,
            "test_end_exclusive": config.selection_config.n_points,
            "fit_scope": "first 50 percent only; validation is not included",
            "row_split_rule": "all target endpoints must lie within the same split",
            "model_unit": "one model per mechanism, seed and representation",
            "test_materialization_supported": True,
            "tuning_supported": False,
        },
    }


def _environment(
    *,
    config: TestConfig,
    run_id: str,
    started: datetime,
    git_commit: str,
    git_dirty: bool,
    command_args: Sequence[str],
    train_end: int,
    test_start: int,
    mode: str,
    primary_dir: Path | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "running",
        "started_utc": started.isoformat(),
        "phase": config.phase,
        "scope": config.scope,
        "mode": mode,
        "replication_of": str(primary_dir) if primary_dir is not None else None,
        "decisory": mode == "primary",
        "test_opened": False,
        "test_materialized": False,
        "selection_lock_sha256": config.selection_lock_sha256,
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
            "vectorchain": validation._package_version("vectorchain"),
            "numpy": np.__version__,
            "matplotlib": validation._package_version("matplotlib"),
        },
        "config": _resolved_config(config, train_end, test_start),
        "command": {"argv": list(command_args), "display": " ".join(command_args)},
    }


def _validate_selection_lock(payload: Mapping[str, Any], config_hash: str) -> None:
    try:
        source = payload["source"]
        selected = payload["selected"]
        valid = (
            payload["status"] == "canonical_selection_train_only"
            and payload["decisory"] is False
            and payload["test_materialized"] is False
            and source["git_dirty"] is False
            and source["generated_stop_exclusive"] == 2867
            and source["config_sha256"] == config_hash
            and selected["lambda_revision"] == 0.1
            and selected["lambda_bend"] == 1.0
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        msg = "selection lock is not the frozen canonical train-only result"
        raise ValueError(msg)


def _selected_penalties(selection: Mapping[str, Any]) -> tuple[float, float]:
    selected = selection["selected"]
    return float(selected["lambda_revision"]), float(selected["lambda_bend"])


def _validate_canonical_dimensions(
    mechanisms: Sequence[str], seeds: Sequence[int], horizons: Sequence[int]
) -> None:
    if tuple(mechanisms) != tuple(sorted(revisable_chain.MECHANISM_NAMES)):
        msg = "gate requires all three registered mechanisms"
        raise ValueError(msg)
    if tuple(seeds) != tuple(sorted(validation.CANONICAL_SELECTION_SEEDS)):
        msg = "gate requires all five registered canonical seeds"
        raise ValueError(msg)
    if tuple(horizons) != tuple(sorted(revisable_chain.HORIZONS)):
        msg = "gate requires all three registered horizons"
        raise ValueError(msg)


def _row_split(row_index: int, train_mask: NDArray[np.bool_], test_mask: NDArray[np.bool_]) -> str:
    if train_mask[row_index]:
        return "train"
    if test_mask[row_index]:
        return "test"
    return "validation_or_boundary_excluded"


def _geometric_mean(values: Sequence[float] | NDArray[np.float64] | Any) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        msg = "geometric mean requires non-empty finite non-negative values"
        raise ValueError(msg)
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or numerator < 0.0:
        msg = "ratio numerator must be finite and non-negative"
        raise ValueError(msg)
    if not np.isfinite(denominator) or denominator <= 0.0:
        msg = "ratio denominator must be finite and positive"
        raise ValueError(msg)
    return numerator / denominator


def _csv_without_columns(path: Path, excluded: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: value for key, value in row.items() if key not in excluded}
            for row in csv.DictReader(stream)
        ]


def _relative_file(config_path: Path, value: object, name: str) -> Path:
    relative = _non_empty_string(value, name)
    candidate = (config_path.parent / relative).resolve()
    if not candidate.is_file():
        msg = f"{name} does not reference an existing file"
        raise ValueError(msg)
    return candidate


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


def _sha256_string(value: object, name: str) -> str:
    result = _non_empty_string(value, name).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        msg = f"{name} must be a lowercase SHA-256 hex digest"
        raise ValueError(msg)
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{name} must be a boolean"
        raise TypeError(msg)
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be greater than or equal to {minimum}"
        raise ValueError(msg)
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a real number"
        raise TypeError(msg)
    result = float(value)
    if not np.isfinite(result):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return result


def _require_exact_real(table: Mapping[str, Any], key: str, expected: float) -> None:
    if _real(table.get(key), key) != expected:
        msg = f"{key} must equal the pre-registered value {expected}"
        raise ValueError(msg)


def _require_exact_integer(table: Mapping[str, Any], key: str, expected: int) -> None:
    if _integer(table.get(key), key, minimum=0) != expected:
        msg = f"{key} must equal the pre-registered value {expected}"
        raise ValueError(msg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open or reproduce the lock-bound canonical K7 test exactly once"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--replicate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one-way K7 test command without treating a negative gate as a crash."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("10_revisable_chain_test.py", *argv))
    result = run_test(
        arguments.config,
        output_root=arguments.output_root,
        replication_of=arguments.replicate,
        command_args=command_args,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "mode": result.mode,
                "gate_passed": result.gate_passed,
                "execution_complete": result.n_failures == 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
