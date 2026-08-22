"""Reproducible similarity, ablation, and nearest-neighbor benchmark."""

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
    FeatureStandardizer,
    VectorChain,
    dtw_distance,
    first_difference,
    first_second_difference,
    fixed_linear_segmentation,
    generate_chirp,
    generate_first_order_response,
    generate_piecewise_linear,
    generate_ramp,
    generate_regime_change,
    generate_second_order_response,
    generate_sine,
    normalized_raw_values,
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
    "sine": frozenset({"frequency", "phase"}),
    "chirp": frozenset({"start_frequency", "end_frequency", "phase"}),
    "ramp": frozenset(),
    "piecewise_linear": frozenset(),
    "first_order_response": frozenset({"time_constant"}),
    "second_order_response": frozenset({"natural_frequency", "damping_ratio"}),
    "regime_change": frozenset(
        {
            "frequency_before",
            "frequency_after",
            "change_fraction",
            "level_shift",
        }
    ),
}

METRIC_FIELDS = (
    "representation",
    "family",
    "feature_names",
    "status",
    "error_type",
    "error_message",
    "n_features",
    "mean_gallery_length",
    "mean_query_length",
    "top_1_accuracy",
    "top_3_accuracy",
    "mean_reciprocal_rank",
    "mean_within_class_distance",
    "mean_between_class_distance",
    "separation_ratio",
    "representation_runtime_s",
    "distance_runtime_s",
    "dtw_window",
)

SAMPLE_FIELDS = (
    "sample_id",
    "split",
    "label",
    "variant_index",
    "seed",
    "amplitude",
    "offset",
    "noise_std",
    "n_points",
)

SEQUENCE_FIELDS = (
    "representation",
    "sample_id",
    "split",
    "label",
    "n_steps",
    "n_features",
)

NEIGHBOR_FIELDS = (
    "representation",
    "query_sample_id",
    "query_label",
    "rank",
    "gallery_sample_id",
    "gallery_label",
    "distance",
    "correct",
)


@dataclass(frozen=True, slots=True)
class Variant:
    """Nuisance-factor values for one gallery or query realization."""

    amplitude: float
    offset: float
    noise_std: float


@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    """Validated effective inputs for one similarity benchmark."""

    name: str
    seed: int
    signal_names: tuple[str, ...]
    n_points: int
    gallery_variants: tuple[Variant, ...]
    query_variants: tuple[Variant, ...]
    signal_parameters: Mapping[str, Mapping[str, float]]
    fixed_segment_length: int
    fixed_features: tuple[FeatureName, ...]
    vectorchain_tolerance: float
    vectorchain_min_segment_length: int
    vectorchain_ablations: tuple[tuple[FeatureName, ...], ...]
    dtw_window_fraction: float
    k_values: tuple[int, ...]
    output_root: str
    save_distances: bool
    save_plots: bool
    plot_dpi: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Sample:
    """One immutable labeled realization used by every representation."""

    sample_id: str
    split: str
    label: str
    variant_index: int
    seed: int
    variant: Variant
    values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Representation:
    """One pre-specified transform in the benchmark comparison."""

    name: str
    family: str
    feature_names: tuple[str, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Identity and primary outputs of a completed similarity run."""

    run_id: str
    run_dir: Path
    metrics_path: Path
    n_conditions: int
    n_failures: int


def load_config(path: Path) -> SimilarityConfig:
    """Load and validate a similarity benchmark TOML configuration."""

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    experiment = _table(raw, "experiment")
    dataset = _table(raw, "dataset")
    signals = _table(raw, "signals")
    representations = _table(raw, "representations")
    distance = _table(raw, "distance")
    retrieval = _table(raw, "retrieval")
    output = _table(raw, "output")

    name = _non_empty_string(experiment.get("name"), "experiment.name")
    seed = _integer(experiment.get("seed"), "experiment.seed", minimum=0)
    signal_names = _unique_string_tuple(dataset.get("signal_names"), "dataset.signal_names")
    unknown = tuple(signal for signal in signal_names if signal not in SIGNAL_GENERATORS)
    if unknown:
        msg = f"unsupported signals: {unknown}"
        raise ValueError(msg)
    n_points = _integer(dataset.get("n_points"), "dataset.n_points", minimum=3)
    gallery_variants = _variants(dataset.get("gallery_variants"), "dataset.gallery_variants")
    query_variants = _variants(dataset.get("query_variants"), "dataset.query_variants")
    overlap = set(gallery_variants) & set(query_variants)
    if overlap:
        msg = "gallery and query variants must be disjoint"
        raise ValueError(msg)

    signal_parameters = _signal_parameter_tables(signals.get("parameters"), signal_names)
    fixed_segment_length = _integer(
        representations.get("fixed_segment_length"),
        "representations.fixed_segment_length",
        minimum=1,
    )
    fixed_features = validate_feature_names(
        _unique_string_tuple(
            representations.get("fixed_features"), "representations.fixed_features"
        )
    )
    vectorchain_tolerance = _real(
        representations.get("vectorchain_tolerance"),
        "representations.vectorchain_tolerance",
        minimum=0.0,
    )
    vectorchain_min_segment_length = _integer(
        representations.get("vectorchain_min_segment_length"),
        "representations.vectorchain_min_segment_length",
        minimum=2,
    )
    vectorchain_ablations = _feature_ablations(representations.get("vectorchain_ablations"))
    for features in vectorchain_ablations:
        VectorChain(
            tolerance=vectorchain_tolerance,
            causal=True,
            min_segment_length=vectorchain_min_segment_length,
            features=features,
        )

    dtw_window_fraction = _real(
        distance.get("dtw_window_fraction"), "distance.dtw_window_fraction", minimum=0.0
    )
    if dtw_window_fraction > 1.0:
        msg = "distance.dtw_window_fraction must be less than or equal to 1"
        raise ValueError(msg)
    k_values = _integer_tuple(retrieval.get("k_values"), "retrieval.k_values", minimum=1)
    if k_values != (1, 3):
        msg = "retrieval.k_values must exactly equal [1, 3] for the registered protocol"
        raise ValueError(msg)
    gallery_size = len(signal_names) * len(gallery_variants)
    if gallery_size < max(k_values):
        msg = "gallery must contain at least max(retrieval.k_values) samples"
        raise ValueError(msg)

    output_root = _non_empty_string(output.get("root"), "output.root")
    save_distances = _boolean(output.get("save_distances"), "output.save_distances")
    save_plots = _boolean(output.get("save_plots"), "output.save_plots")
    plot_dpi = _integer(output.get("plot_dpi"), "output.plot_dpi", minimum=72)

    return SimilarityConfig(
        name=name,
        seed=seed,
        signal_names=signal_names,
        n_points=n_points,
        gallery_variants=gallery_variants,
        query_variants=query_variants,
        signal_parameters=signal_parameters,
        fixed_segment_length=fixed_segment_length,
        fixed_features=fixed_features,
        vectorchain_tolerance=vectorchain_tolerance,
        vectorchain_min_segment_length=vectorchain_min_segment_length,
        vectorchain_ablations=vectorchain_ablations,
        dtw_window_fraction=dtw_window_fraction,
        k_values=k_values,
        output_root=output_root,
        save_distances=save_distances,
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
    """Execute every representation on one shared immutable dataset split."""

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

    samples = _build_samples(config)
    gallery = tuple(sample for sample in samples if sample.split == "gallery")
    queries = tuple(sample for sample in samples if sample.split == "query")
    representations = _representations(config)
    sample_seeds = {sample.sample_id: sample.seed for sample in samples}

    effective_config = dict(config.raw)
    effective_config["resolved"] = {
        "config_path": _display_path(config_path, repository_root),
        "config_sha256": config_digest,
        "output_root": str(resolved_root),
        "sample_seeds": sample_seeds,
        "representations": [representation.name for representation in representations],
        "gallery_sample_ids": [sample.sample_id for sample in gallery],
        "query_sample_ids": [sample.sample_id for sample in queries],
    }
    _write_json(run_dir / "config.json", effective_config)
    environment = _environment_manifest(
        run_id=run_id,
        started=started,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=effective_config,
        sample_seeds=sample_seeds,
        command_args=tuple(command_args if command_args is not None else sys.argv),
    )
    _write_json(run_dir / "environment.json", environment)
    _write_csv(run_dir / "samples.csv", SAMPLE_FIELDS, _sample_rows(samples))

    metrics_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    distance_matrices: list[NDArray[np.float64]] = []
    successful_names: list[str] = []
    run_start = time.perf_counter_ns()
    try:
        for representation in representations:
            try:
                result = _run_representation(representation, gallery, queries, config)
                metrics_rows.append(result[0])
                sequence_rows.extend(result[1])
                neighbor_rows.extend(result[2])
                distance_matrices.append(result[3])
                successful_names.append(representation.name)
            except Exception as error:
                metrics_rows.append(_failure_row(representation, error))
            _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metrics_rows)
            _write_csv(run_dir / "sequences.csv", SEQUENCE_FIELDS, sequence_rows)
            _write_csv(run_dir / "neighbors.csv", NEIGHBOR_FIELDS, neighbor_rows)

        if config.save_distances and distance_matrices:
            np.savez_compressed(
                run_dir / "distances.npz",
                representation_names=np.asarray(successful_names),
                query_sample_ids=np.asarray([sample.sample_id for sample in queries]),
                gallery_sample_ids=np.asarray([sample.sample_id for sample in gallery]),
                distances=np.stack(distance_matrices),
            )
        if config.save_plots:
            _plot_summaries(metrics_rows, plots_dir, run_id, config.plot_dpi)
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


def _build_samples(config: SimilarityConfig) -> tuple[Sample, ...]:
    samples: list[Sample] = []
    for split, variants in (
        ("gallery", config.gallery_variants),
        ("query", config.query_variants),
    ):
        for label in config.signal_names:
            for variant_index, variant in enumerate(variants):
                sample_id = f"{split}__{label}__v{variant_index}"
                seed = _derive_seed(config.seed, sample_id)
                values = SIGNAL_GENERATORS[label](
                    rng=seed,
                    n_points=config.n_points,
                    amplitude=variant.amplitude,
                    offset=variant.offset,
                    noise_std=variant.noise_std,
                    **config.signal_parameters[label],
                )
                values.flags.writeable = False
                samples.append(
                    Sample(sample_id, split, label, variant_index, seed, variant, values)
                )
    return tuple(samples)


def _representations(config: SimilarityConfig) -> tuple[Representation, ...]:
    baseline = (
        Representation("raw", "baseline", ("value",), "raw"),
        Representation("normalized_raw", "baseline", ("value",), "normalized_raw"),
        Representation("first_difference", "baseline", ("delta_y",), "first_difference"),
        Representation(
            "first_second_difference",
            "baseline",
            ("delta_y", "delta2_y"),
            "first_second_difference",
        ),
        Representation("fixed_linear", "baseline", config.fixed_features, "fixed_linear"),
    )
    ablations = tuple(
        Representation(
            f"vectorchain__{'-'.join(features)}",
            "vectorchain",
            features,
            "vectorchain",
        )
        for features in config.vectorchain_ablations
    )
    return baseline + ablations


def _run_representation(
    representation: Representation,
    gallery: Sequence[Sample],
    queries: Sequence[Sample],
    config: SimilarityConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    NDArray[np.float64],
]:
    representation_start = time.perf_counter_ns()
    gallery_sequences = tuple(
        _transform(representation, sample.values, config) for sample in gallery
    )
    query_sequences = tuple(_transform(representation, sample.values, config) for sample in queries)
    standardizer = FeatureStandardizer.fit(gallery_sequences)
    scaled_gallery = tuple(standardizer.transform(sequence) for sequence in gallery_sequences)
    scaled_queries = tuple(standardizer.transform(sequence) for sequence in query_sequences)
    representation_runtime_s = (time.perf_counter_ns() - representation_start) / 1e9

    maximum_length = max(sequence.shape[0] for sequence in (*scaled_gallery, *scaled_queries))
    window = math.ceil(maximum_length * config.dtw_window_fraction)
    distance_start = time.perf_counter_ns()
    distances = np.empty((len(queries), len(gallery)), dtype=np.float64)
    for query_index, query in enumerate(scaled_queries):
        for gallery_index, reference in enumerate(scaled_gallery):
            distances[query_index, gallery_index] = dtw_distance(query, reference, window=window)
    distance_runtime_s = (time.perf_counter_ns() - distance_start) / 1e9

    rankings = np.argsort(distances, axis=1, kind="stable")
    gallery_labels = np.asarray([sample.label for sample in gallery])
    query_labels = np.asarray([sample.label for sample in queries])
    ranked_labels = gallery_labels[rankings]
    correct = ranked_labels == query_labels[:, np.newaxis]
    reciprocal_ranks = 1.0 / (np.argmax(correct, axis=1) + 1)
    same_class = query_labels[:, np.newaxis] == gallery_labels[np.newaxis, :]
    within = float(np.mean(distances[same_class]))
    between = float(np.mean(distances[~same_class]))

    metrics: dict[str, object] = {
        "representation": representation.name,
        "family": representation.family,
        "feature_names": ",".join(representation.feature_names),
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "n_features": scaled_gallery[0].shape[1],
        "mean_gallery_length": float(np.mean([sequence.shape[0] for sequence in scaled_gallery])),
        "mean_query_length": float(np.mean([sequence.shape[0] for sequence in scaled_queries])),
        "top_1_accuracy": float(np.mean(correct[:, :1].any(axis=1))),
        "top_3_accuracy": float(np.mean(correct[:, :3].any(axis=1))),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "mean_within_class_distance": within,
        "mean_between_class_distance": between,
        "separation_ratio": between / within if within > 0.0 else float("inf"),
        "representation_runtime_s": representation_runtime_s,
        "distance_runtime_s": distance_runtime_s,
        "dtw_window": window,
    }

    sequence_rows = [
        {
            "representation": representation.name,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "label": sample.label,
            "n_steps": sequence.shape[0],
            "n_features": sequence.shape[1],
        }
        for sample, sequence in (
            *(
                (sample, sequence)
                for sample, sequence in zip(gallery, gallery_sequences, strict=True)
            ),
            *(
                (sample, sequence)
                for sample, sequence in zip(queries, query_sequences, strict=True)
            ),
        )
    ]
    neighbor_rows: list[dict[str, object]] = []
    for query_index, query_sample in enumerate(queries):
        for rank, gallery_index in enumerate(rankings[query_index], start=1):
            gallery_sample = gallery[int(gallery_index)]
            neighbor_rows.append(
                {
                    "representation": representation.name,
                    "query_sample_id": query_sample.sample_id,
                    "query_label": query_sample.label,
                    "rank": rank,
                    "gallery_sample_id": gallery_sample.sample_id,
                    "gallery_label": gallery_sample.label,
                    "distance": distances[query_index, gallery_index],
                    "correct": gallery_sample.label == query_sample.label,
                }
            )
    return metrics, sequence_rows, neighbor_rows, distances


def _transform(
    representation: Representation,
    values: NDArray[np.float64],
    config: SimilarityConfig,
) -> NDArray[np.float64]:
    if representation.kind == "raw":
        return raw_values(values)
    if representation.kind == "normalized_raw":
        return normalized_raw_values(values)
    if representation.kind == "first_difference":
        return first_difference(values)
    if representation.kind == "first_second_difference":
        return first_second_difference(values)
    if representation.kind == "fixed_linear":
        return fixed_linear_segmentation(
            values,
            segment_length=config.fixed_segment_length,
            features=config.fixed_features,
        ).vectors
    if representation.kind == "vectorchain":
        chain = VectorChain(
            tolerance=config.vectorchain_tolerance,
            causal=True,
            min_segment_length=config.vectorchain_min_segment_length,
            features=representation.feature_names,
        )
        chain.reset()
        for value in values:
            chain.update(float(value))
        chain.finalize()
        return chain.vectors_.copy()
    msg = f"unsupported representation kind: {representation.kind}"
    raise ValueError(msg)


def _sample_rows(samples: Sequence[Sample]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "label": sample.label,
            "variant_index": sample.variant_index,
            "seed": sample.seed,
            "amplitude": sample.variant.amplitude,
            "offset": sample.variant.offset,
            "noise_std": sample.variant.noise_std,
            "n_points": sample.values.size,
        }
        for sample in samples
    ]


def _failure_row(representation: Representation, error: Exception) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in METRIC_FIELDS}
    row.update(
        {
            "representation": representation.name,
            "family": representation.family,
            "feature_names": ",".join(representation.feature_names),
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    return row


def _plot_summaries(
    rows: Sequence[Mapping[str, object]], plots_dir: Path, run_id: str, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    successful = [row for row in rows if row["status"] == "ok"]
    labels = [_short_label(str(row["representation"])) for row in successful]
    positions = np.arange(len(successful))

    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    width = 0.38
    axis.barh(
        positions - width / 2,
        [float(row["top_1_accuracy"]) for row in successful],
        height=width,
        label="top-1",
    )
    axis.barh(
        positions + width / 2,
        [float(row["top_3_accuracy"]) for row in successful],
        height=width,
        label="top-3",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Accuracy")
    axis.set_title(f"Retrieval accuracy by representation\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__retrieval-accuracy.png", dpi=dpi)
    plt.close(figure)

    _horizontal_metric_plot(
        successful,
        labels,
        "separation_ratio",
        "Between-class / within-class distance",
        "Class separation by representation",
        plots_dir / "summary__separation-ratio.png",
        run_id,
        dpi,
    )

    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    axis.barh(
        positions - width / 2,
        [float(row["mean_gallery_length"]) for row in successful],
        height=width,
        label="gallery",
    )
    axis.barh(
        positions + width / 2,
        [float(row["mean_query_length"]) for row in successful],
        height=width,
        label="query",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Mean sequence length")
    axis.set_title(f"Representation length\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.savefig(plots_dir / "summary__sequence-length.png", dpi=dpi)
    plt.close(figure)

    ablations = [row for row in successful if row["family"] == "vectorchain"]
    ablation_labels = [_short_label(str(row["representation"])) for row in ablations]
    _horizontal_metric_plot(
        ablations,
        ablation_labels,
        "top_1_accuracy",
        "Top-1 accuracy",
        "VectorChain feature ablations",
        plots_dir / "summary__vectorchain-ablations.png",
        run_id,
        dpi,
        xlim=(0.0, 1.0),
    )


def _horizontal_metric_plot(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    metric: str,
    xlabel: str,
    title: str,
    path: Path,
    run_id: str,
    dpi: int,
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    axis.barh(np.arange(len(rows)), [float(row[metric]) for row in rows])
    axis.set_yticks(np.arange(len(rows)), labels)
    axis.set_xlabel(xlabel)
    axis.set_title(f"{title}\nrun={run_id}", fontsize=10.0)
    axis.grid(axis="x", alpha=0.25)
    if xlim is not None:
        axis.set_xlim(*xlim)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _short_label(name: str) -> str:
    return name.replace("vectorchain__", "VC: ").replace("_", " ")


def _derive_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
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
    sample_seeds: Mapping[str, int],
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
        "seeds": {"base": config["experiment"]["seed"], "samples": dict(sample_seeds)},
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


def _variants(value: object, name: str) -> tuple[Variant, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty array of tables"
        raise ValueError(msg)
    variants: list[Variant] = []
    expected = {"amplitude", "offset", "noise_std"}
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != expected:
            msg = f"{name}[{index}] must contain exactly {tuple(sorted(expected))}"
            raise ValueError(msg)
        variants.append(
            Variant(
                amplitude=_real(item["amplitude"], f"{name}[{index}].amplitude"),
                offset=_real(item["offset"], f"{name}[{index}].offset"),
                noise_std=_real(item["noise_std"], f"{name}[{index}].noise_std", minimum=0.0),
            )
        )
    result = tuple(variants)
    if len(set(result)) != len(result):
        msg = f"{name} must not contain duplicate variants"
        raise ValueError(msg)
    return result


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


def _feature_ablations(value: object) -> tuple[tuple[FeatureName, ...], ...]:
    if not isinstance(value, list) or not value:
        msg = "representations.vectorchain_ablations must be a non-empty list"
        raise ValueError(msg)
    ablations: list[tuple[FeatureName, ...]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list):
            msg = f"representations.vectorchain_ablations[{index}] must be a list"
            raise ValueError(msg)
        ablations.append(validate_feature_names(item))
    result = tuple(ablations)
    if len(set(result)) != len(result):
        msg = "representations.vectorchain_ablations must not contain duplicates"
        raise ValueError(msg)
    return result


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


def _integer_tuple(value: object, name: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{name} must be a non-empty list of integers"
        raise ValueError(msg)
    result = tuple(_integer(item, name, minimum=minimum) for item in value)
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
        default=Path("configs/similarity/baseline.toml"),
        help="TOML similarity benchmark configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured artifact root (primarily for tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line similarity benchmark entry point."""

    arguments = _build_parser().parse_args(argv)
    command_args = tuple(sys.argv if argv is None else ("02_similarity.py", *argv))
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
