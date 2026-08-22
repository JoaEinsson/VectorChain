"""Configurable reconstruction-compression benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
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
    compression_factor,
    generate_chirp,
    generate_first_order_response,
    generate_piecewise_linear,
    generate_ramp,
    generate_regime_change,
    generate_second_order_response,
    generate_sine,
    mae,
    retention_fraction,
    rmse,
)
from vectorchain.plotting import plot_vector_chain

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

METRIC_NAMES = (
    "n_points",
    "n_vectors",
    "compression_factor",
    "retention_fraction",
    "mae",
    "rmse",
    "transform_runtime_median_s",
    "transform_runtime_q1_s",
    "transform_runtime_q3_s",
    "reconstruction_runtime_median_s",
    "reconstruction_runtime_q1_s",
    "reconstruction_runtime_q3_s",
)

METRIC_FIELDS = (
    "signal",
    "signal_seed",
    "tolerance",
    "status",
    "error_type",
    "error_message",
    *METRIC_NAMES,
)

TIMING_FIELDS = ("signal", "signal_seed", "tolerance", "phase", "repetition", "duration_s")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated effective inputs for one benchmark run."""

    name: str
    seed: int
    repetitions: int
    warmup_repetitions: int
    signal_names: tuple[str, ...]
    n_points: int
    noise_std: float
    signal_parameters: Mapping[str, Mapping[str, float]]
    causal: bool
    min_segment_length: int
    features: tuple[str, ...]
    tolerances: tuple[float, ...]
    metric_names: tuple[str, ...]
    output_root: str
    save_vectors: bool
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of a completed benchmark run."""

    run_id: str
    run_dir: Path
    metrics_path: Path
    n_conditions: int
    n_failures: int


def load_config(path: Path) -> BenchmarkConfig:
    """Load and validate a benchmark TOML configuration."""

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    experiment = _table(raw, "experiment")
    signals = _table(raw, "signals")
    vectorchain = _table(raw, "vectorchain")
    metrics = _table(raw, "metrics")
    output = _table(raw, "output")

    name = _non_empty_string(experiment.get("name"), "experiment.name")
    seed = _integer(experiment.get("seed"), "experiment.seed", minimum=0)
    repetitions = _integer(experiment.get("repetitions"), "experiment.repetitions", minimum=1)
    warmups = _integer(
        experiment.get("warmup_repetitions"),
        "experiment.warmup_repetitions",
        minimum=0,
    )

    signal_names = _unique_string_tuple(signals.get("names"), "signals.names")
    unknown_signals = tuple(name for name in signal_names if name not in SIGNAL_GENERATORS)
    if unknown_signals:
        msg = f"unsupported signals: {unknown_signals}"
        raise ValueError(msg)
    n_points = _integer(signals.get("n_points"), "signals.n_points", minimum=2)
    noise_std = _real(signals.get("noise_std"), "signals.noise_std", minimum=0.0)
    parameters = _signal_parameter_tables(signals.get("parameters"), signal_names)

    causal = _boolean(vectorchain.get("causal"), "vectorchain.causal")
    min_segment_length = _integer(
        vectorchain.get("min_segment_length"),
        "vectorchain.min_segment_length",
        minimum=2,
    )
    features = _unique_string_tuple(vectorchain.get("features"), "vectorchain.features")
    tolerances = _strictly_increasing_reals(
        vectorchain.get("tolerances"), "vectorchain.tolerances", minimum=0.0
    )
    VectorChain(
        tolerance=tolerances[0],
        causal=causal,
        min_segment_length=min_segment_length,
        features=features,
    )

    metric_names = _unique_string_tuple(metrics.get("names"), "metrics.names")
    if metric_names != METRIC_NAMES:
        msg = f"metrics.names must exactly equal {METRIC_NAMES}"
        raise ValueError(msg)

    output_root = _non_empty_string(output.get("root"), "output.root")
    save_vectors = _boolean(output.get("save_vectors"), "output.save_vectors")
    save_plots = _boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = _integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    return BenchmarkConfig(
        name=name,
        seed=seed,
        repetitions=repetitions,
        warmup_repetitions=warmups,
        signal_names=signal_names,
        n_points=n_points,
        noise_std=noise_std,
        signal_parameters=parameters,
        causal=causal,
        min_segment_length=min_segment_length,
        features=features,
        tolerances=tolerances,
        metric_names=metric_names,
        output_root=output_root,
        save_vectors=save_vectors,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
        raw=raw,
    )


def run_experiment(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] | None = None,
) -> RunSummary:
    """Execute the benchmark and persist a complete immutable run directory."""

    config_path = config_path.resolve()
    config = load_config(config_path)
    if config.save_plots:
        import matplotlib

        matplotlib.use("Agg")
    repository_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = _git_state(repository_root)
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    started = datetime.now(UTC)
    timestamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_{config_digest[:8]}_{git_commit[:7]}"
    resolved_root = (
        output_root.resolve()
        if output_root is not None
        else (repository_root / config.output_root).resolve()
    )
    run_dir = resolved_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    vectors_dir = run_dir / "vectors"
    plots_dir = run_dir / "plots"
    if config.save_vectors:
        vectors_dir.mkdir()
    if config.save_plots:
        plots_dir.mkdir()

    seeds = {name: _derive_seed(config.seed, name) for name in config.signal_names}
    effective_config = dict(config.raw)
    effective_config["resolved"] = {
        "config_path": _display_path(config_path, repository_root),
        "config_sha256": config_digest,
        "signal_seeds": seeds,
        "output_root": str(resolved_root),
    }
    _write_json(run_dir / "config.json", effective_config)

    environment = _environment_manifest(
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=effective_config,
        seeds=seeds,
        command_args=tuple(command_args if command_args is not None else sys.argv),
    )
    _write_json(run_dir / "environment.json", environment)

    metrics_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    run_start = time.perf_counter_ns()
    try:
        for signal_name in config.signal_names:
            signal_seed = seeds[signal_name]
            try:
                signal = SIGNAL_GENERATORS[signal_name](
                    rng=signal_seed,
                    n_points=config.n_points,
                    noise_std=config.noise_std,
                    **config.signal_parameters[signal_name],
                )
            except Exception as error:
                for tolerance in config.tolerances:
                    metrics_rows.append(
                        _failure_row(signal_name, signal_seed, tolerance, error, config.n_points)
                    )
                _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metrics_rows)
                continue

            for tolerance in config.tolerances:
                try:
                    row, condition_timings, chain, reconstructed = _run_condition(
                        signal_name,
                        signal_seed,
                        signal,
                        tolerance,
                        config,
                    )
                    timing_rows.extend(condition_timings)
                    condition_name = f"{signal_name}__tol-{_number_tag(tolerance)}"
                    if config.save_vectors:
                        np.savez_compressed(
                            vectors_dir / f"{condition_name}.npz",
                            original=signal,
                            reconstructed=reconstructed,
                            vectors=chain.vectors_,
                            boundaries=chain.segment_boundaries_,
                            feature_names=np.asarray(chain.feature_names_),
                        )
                    if config.save_plots:
                        title = (
                            f"{signal_name} | seed={signal_seed} | tolerance={tolerance:g} | "
                            f"n={config.n_points} | run={run_id}"
                        )
                        axis = plot_vector_chain(signal, chain, title=title)
                        axis.figure.savefig(
                            plots_dir / f"chain__{condition_name}.png",
                            dpi=config.plot_dpi,
                        )
                        _close_figure(axis.figure)
                    metrics_rows.append(row)
                except Exception as error:
                    metrics_rows.append(
                        _failure_row(signal_name, signal_seed, tolerance, error, signal.size)
                    )
                _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metrics_rows)
                _write_csv(run_dir / "timings.csv", TIMING_FIELDS, timing_rows)

        if config.save_plots:
            _plot_summaries(metrics_rows, config, plots_dir, run_id)
        _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metrics_rows)
        _write_csv(run_dir / "timings.csv", TIMING_FIELDS, timing_rows)
    except Exception as error:
        environment["status"] = "failed"
        environment["failure"] = {"type": type(error).__name__, "message": str(error)}
        environment["finished_utc"] = datetime.now(UTC).isoformat()
        environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
        _write_json(run_dir / "environment.json", environment)
        _write_manifest(run_dir, status="failed")
        raise

    n_failures = sum(row["status"] != "ok" for row in metrics_rows)
    environment["status"] = "complete" if n_failures == 0 else "complete_with_failures"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - run_start) / 1e9
    environment["n_conditions"] = len(metrics_rows)
    environment["n_failures"] = n_failures
    _write_json(run_dir / "environment.json", environment)
    _write_manifest(run_dir, status=str(environment["status"]))
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.csv",
        n_conditions=len(metrics_rows),
        n_failures=n_failures,
    )


def _run_condition(
    signal_name: str,
    signal_seed: int,
    signal: NDArray[np.float64],
    tolerance: float,
    config: BenchmarkConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    VectorChain,
    NDArray[np.float64],
]:
    for _ in range(config.warmup_repetitions):
        warmup = _new_chain(config, tolerance)
        warmup_vectors = _transform_signal(warmup, signal)
        warmup.inverse_transform(warmup_vectors)

    timing_rows: list[dict[str, object]] = []
    transform_durations: list[float] = []
    reconstruction_durations: list[float] = []
    expected_vectors: NDArray[np.float64] | None = None
    expected_boundaries: NDArray[np.int64] | None = None
    expected_reconstruction: NDArray[np.float64] | None = None
    final_chain: VectorChain | None = None

    for repetition in range(config.repetitions):
        chain = _new_chain(config, tolerance)
        started = time.perf_counter_ns()
        vectors = _transform_signal(chain, signal)
        transform_s = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        reconstructed = chain.inverse_transform(vectors)
        reconstruction_s = (time.perf_counter_ns() - started) / 1e9

        if expected_vectors is None:
            expected_vectors = vectors.copy()
            expected_boundaries = chain.segment_boundaries_.copy()
            expected_reconstruction = reconstructed.copy()
        elif not (
            np.array_equal(vectors, expected_vectors)
            and np.array_equal(chain.segment_boundaries_, expected_boundaries)
            and np.array_equal(reconstructed, expected_reconstruction)
        ):
            msg = "deterministic outputs changed between timing repetitions"
            raise RuntimeError(msg)

        transform_durations.append(transform_s)
        reconstruction_durations.append(reconstruction_s)
        timing_rows.extend(
            (
                {
                    "signal": signal_name,
                    "signal_seed": signal_seed,
                    "tolerance": tolerance,
                    "phase": "transform",
                    "repetition": repetition,
                    "duration_s": transform_s,
                },
                {
                    "signal": signal_name,
                    "signal_seed": signal_seed,
                    "tolerance": tolerance,
                    "phase": "reconstruction",
                    "repetition": repetition,
                    "duration_s": reconstruction_s,
                },
            )
        )
        final_chain = chain

    if final_chain is None or expected_reconstruction is None:
        msg = "at least one timing repetition is required"
        raise RuntimeError(msg)
    transform_q1, transform_median, transform_q3 = _quartiles(transform_durations)
    reconstruction_q1, reconstruction_median, reconstruction_q3 = _quartiles(
        reconstruction_durations
    )
    row: dict[str, object] = {
        "signal": signal_name,
        "signal_seed": signal_seed,
        "tolerance": tolerance,
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "n_points": signal.size,
        "n_vectors": final_chain.vectors_.shape[0],
        "compression_factor": compression_factor(signal.size, final_chain.vectors_.shape[0]),
        "retention_fraction": retention_fraction(signal.size, final_chain.vectors_.shape[0]),
        "mae": mae(signal, expected_reconstruction),
        "rmse": rmse(signal, expected_reconstruction),
        "transform_runtime_median_s": transform_median,
        "transform_runtime_q1_s": transform_q1,
        "transform_runtime_q3_s": transform_q3,
        "reconstruction_runtime_median_s": reconstruction_median,
        "reconstruction_runtime_q1_s": reconstruction_q1,
        "reconstruction_runtime_q3_s": reconstruction_q3,
    }
    return row, timing_rows, final_chain, expected_reconstruction


def _new_chain(config: BenchmarkConfig, tolerance: float) -> VectorChain:
    return VectorChain(
        tolerance=tolerance,
        causal=config.causal,
        min_segment_length=config.min_segment_length,
        features=config.features,
    )


def _transform_signal(chain: VectorChain, signal: NDArray[np.float64]) -> NDArray[np.float64]:
    chain.reset()
    for value in signal:
        chain.update(float(value))
    chain.finalize()
    return chain.vectors_.copy()


def _plot_summaries(
    metrics_rows: Sequence[Mapping[str, object]],
    config: BenchmarkConfig,
    plots_dir: Path,
    run_id: str,
) -> None:
    import matplotlib.pyplot as plt

    successful = [row for row in metrics_rows if row["status"] == "ok"]
    plot_specs = (
        ("rmse", "RMSE", "summary__rmse-by-tolerance.png"),
        (
            "compression_factor",
            "Structural compression factor (n_points / n_vectors)",
            "summary__compression-by-tolerance.png",
        ),
    )
    for metric, ylabel, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for signal_name in config.signal_names:
            rows = sorted(
                (row for row in successful if row["signal"] == signal_name),
                key=lambda row: float(row["tolerance"]),
            )
            if rows:
                axis.plot(
                    [float(row["tolerance"]) for row in rows],
                    [float(row[metric]) for row in rows],
                    marker="o",
                    label=signal_name,
                )
        if config.tolerances[0] > 0.0:
            axis.set_xscale("log")
        axis.set_xlabel("Absolute tolerance")
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"{ylabel} by tolerance | seed={config.seed} | n={config.n_points} | run={run_id}"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small")
        figure.savefig(plots_dir / filename, dpi=config.plot_dpi)
        _close_figure(figure)

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for signal_name in config.signal_names:
        rows = sorted(
            (row for row in successful if row["signal"] == signal_name),
            key=lambda row: float(row["tolerance"]),
        )
        if rows:
            axis.plot(
                [float(row["compression_factor"]) for row in rows],
                [float(row["rmse"]) for row in rows],
                marker="o",
                label=signal_name,
            )
    axis.set_xlabel("Structural compression factor (n_points / n_vectors)")
    axis.set_ylabel("RMSE")
    axis.set_title(
        f"Compression-reconstruction tradeoff | seed={config.seed} | n={config.n_points} | "
        f"run={run_id}"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(plots_dir / "summary__compression-rmse-tradeoff.png", dpi=config.plot_dpi)
    _close_figure(figure)


def _close_figure(figure: object) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)


def _failure_row(
    signal_name: str,
    signal_seed: int,
    tolerance: float,
    error: Exception,
    n_points: int,
) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "signal": signal_name,
            "signal_seed": signal_seed,
            "tolerance": tolerance,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "n_points": n_points,
        }
    )
    return row


def _quartiles(durations: Sequence[float]) -> tuple[float, float, float]:
    values = np.asarray(durations, dtype=np.float64)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def _derive_seed(base_seed: int, signal_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{signal_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _number_tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


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
    seeds: Mapping[str, int],
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
        "seeds": {"base": config["experiment"]["seed"], "signals": dict(seeds)},
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
    if not isinstance(value, dict):
        msg = "signals.parameters must be a TOML table"
        raise ValueError(msg)
    if set(value) != set(signal_names):
        msg = "signals.parameters must contain exactly one table for every configured signal"
        raise ValueError(msg)

    validated: dict[str, dict[str, float]] = {}
    for signal_name in signal_names:
        parameters = value[signal_name]
        if not isinstance(parameters, dict):
            msg = f"signals.parameters.{signal_name} must be a TOML table"
            raise ValueError(msg)
        if set(parameters) != SIGNAL_PARAMETERS[signal_name]:
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


def _strictly_increasing_reals(
    value: object,
    name: str,
    *,
    minimum: float,
) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty list"
        raise ValueError(msg)
    result = tuple(_real(item, name, minimum=minimum) for item in value)
    if any(right <= left for left, right in itertools.pairwise(result)):
        msg = f"{name} must be strictly increasing"
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
        default=Path("configs/reconstruction/baseline.toml"),
        help="TOML benchmark configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark entry point."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("01_reconstruction.py", *argv))
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
