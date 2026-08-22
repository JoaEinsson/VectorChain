"""Tests for the reproducible similarity and ablation runner."""

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

RUNNER_PATH = Path(__file__).parents[1] / "experiments" / "similarity_experiment.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("vectorchain_similarity_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError("could not load similarity experiment module")
similarity_experiment = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = similarity_experiment
RUNNER_SPEC.loader.exec_module(similarity_experiment)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-similarity-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_config(
    path: Path,
    *,
    save_distances: bool = True,
    save_plots: bool = False,
) -> Path:
    content = f"""
[experiment]
name = "test-similarity"
seed = 2718

[dataset]
signal_names = ["sine", "ramp"]
n_points = 24

[[dataset.gallery_variants]]
amplitude = 0.8
offset = -0.4
noise_std = 0.01

[[dataset.gallery_variants]]
amplitude = 1.2
offset = 0.4
noise_std = 0.02

[[dataset.query_variants]]
amplitude = 1.0
offset = 0.2
noise_std = 0.015

[signals.parameters.sine]
frequency = 2.0
phase = 0.0

[signals.parameters.ramp]

[representations]
fixed_segment_length = 5
fixed_features = ["dt", "dy"]
vectorchain_tolerance = 0.03
vectorchain_min_segment_length = 2
vectorchain_ablations = [["dt", "dy"]]

[distance]
dtw_window_fraction = 0.2

[retrieval]
k_values = [1, 3]

[output]
root = "artifacts"
save_distances = {str(save_distances).lower()}
save_plots = {str(save_plots).lower()}
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_canonical_similarity_config_is_complete_and_valid() -> None:
    config = similarity_experiment.load_config(Path("configs/similarity/baseline.toml"))

    assert config.name == "similarity-retrieval-baseline"
    assert config.signal_names == tuple(similarity_experiment.SIGNAL_GENERATORS)
    assert len(config.gallery_variants) == 3
    assert len(config.query_variants) == 3
    assert len(config.vectorchain_ablations) == 5
    assert config.k_values == (1, 3)
    assert config.save_distances
    assert config.save_plots


def test_runner_uses_one_split_and_persists_complete_audit_trail(
    experiment_path: Path,
) -> None:
    config_path = _write_config(experiment_path / "config.toml", save_plots=True)

    summary = similarity_experiment.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "similarity"),
    )

    assert summary.n_conditions == 6
    assert summary.n_failures == 0
    assert summary.run_dir.parent == experiment_path.resolve() / "runs"
    assert {
        "config.json",
        "distances.npz",
        "environment.json",
        "manifest.json",
        "metrics.csv",
        "neighbors.csv",
        "samples.csv",
        "sequences.csv",
    } <= {path.name for path in summary.run_dir.iterdir()}

    metrics = _read_csv(summary.metrics_path)
    assert len(metrics) == 6
    assert {row["status"] for row in metrics} == {"ok"}
    assert {row["representation"] for row in metrics} == {
        "raw",
        "normalized_raw",
        "first_difference",
        "first_second_difference",
        "fixed_linear",
        "vectorchain__dt-dy",
    }
    assert all(0.0 <= float(row["top_1_accuracy"]) <= 1.0 for row in metrics)
    assert all(0.0 <= float(row["top_3_accuracy"]) <= 1.0 for row in metrics)

    samples = _read_csv(summary.run_dir / "samples.csv")
    assert len(samples) == 6
    assert {row["split"] for row in samples} == {"gallery", "query"}
    assert len({row["seed"] for row in samples}) == 6

    sequences = _read_csv(summary.run_dir / "sequences.csv")
    assert len(sequences) == 36
    sample_ids_by_representation: dict[str, set[str]] = defaultdict(set)
    for row in sequences:
        sample_ids_by_representation[row["representation"]].add(row["sample_id"])
    assert len(sample_ids_by_representation) == 6
    assert len({frozenset(ids) for ids in sample_ids_by_representation.values()}) == 1

    neighbors = _read_csv(summary.run_dir / "neighbors.csv")
    assert len(neighbors) == 48
    assert {int(row["rank"]) for row in neighbors} == {1, 2, 3, 4}

    with np.load(summary.run_dir / "distances.npz") as distances:
        assert distances["distances"].shape == (6, 2, 4)
        assert distances["representation_names"].shape == (6,)

    plot_paths = sorted((summary.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plot_paths} == {
        "summary__retrieval-accuracy.png",
        "summary__separation-ratio.png",
        "summary__sequence-length.png",
        "summary__vectorchain-ablations.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plot_paths)

    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 6
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "similarity"]
    assert len(environment["seeds"]["samples"]) == 6
    assert isinstance(environment["git"]["dirty"], bool)

    manifest = json.loads((summary.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (summary.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_representation_failures_are_preserved_and_reported(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(experiment_path / "failure.toml", save_distances=False)
    original_transform = similarity_experiment._transform

    def fail_raw(representation: object, values: np.ndarray, config: object) -> np.ndarray:
        if representation.name == "raw":
            raise RuntimeError("deliberate representation failure")
        return original_transform(representation, values, config)

    monkeypatch.setattr(similarity_experiment, "_transform", fail_raw)
    summary = similarity_experiment.run_experiment(
        config_path, output_root=experiment_path / "runs"
    )

    assert summary.n_conditions == 6
    assert summary.n_failures == 1
    rows = _read_csv(summary.metrics_path)
    failed = [row for row in rows if row["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["representation"] == "raw"
    assert failed[0]["error_type"] == "RuntimeError"
    assert failed[0]["error_message"] == "deliberate representation failure"
    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete_with_failures"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('signal_names = ["sine", "ramp"]', 'signal_names = ["sine", "sine"]'),
        ("n_points = 24", "n_points = 2"),
        ("noise_std = 0.01", "noise_std = -0.01"),
        ("frequency = 2.0", "unknown = 2.0"),
        ('fixed_features = ["dt", "dy"]', 'fixed_features = ["theta"]'),
        ("dtw_window_fraction = 0.2", "dtw_window_fraction = 1.1"),
        ("k_values = [1, 3]", "k_values = [1, 2]"),
    ],
)
def test_invalid_similarity_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_config(experiment_path / "invalid.toml")
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        similarity_experiment.load_config(config_path)
