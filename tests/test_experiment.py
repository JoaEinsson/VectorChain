"""Tests for the reproducible reconstruction-compression runner."""

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

RUNNER_PATH = Path(__file__).parents[1] / "experiments" / "reconstruction.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "vectorchain_reconstruction_runner", RUNNER_PATH
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError("could not load reconstruction experiment module")
reconstruction = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = reconstruction
RUNNER_SPEC.loader.exec_module(reconstruction)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-experiment-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_config(
    path: Path,
    *,
    save_vectors: bool = True,
    save_plots: bool = False,
    repetitions: int = 2,
    tolerances: tuple[float, ...] = (0.01, 0.03),
) -> Path:
    metric_names = ",\n  ".join(f'"{name}"' for name in reconstruction.METRIC_NAMES)
    tolerance_values = ", ".join(str(value) for value in tolerances)
    content = f"""
[experiment]
name = "test-reconstruction"
seed = 1729
repetitions = {repetitions}
warmup_repetitions = 0

[signals]
names = ["sine"]
n_points = 32
noise_std = 0.01

[signals.parameters.sine]
amplitude = 1.0
offset = 0.0
frequency = 2.0
phase = 0.0

[vectorchain]
causal = true
min_segment_length = 2
features = ["dy", "dt"]
tolerances = [{tolerance_values}]

[metrics]
names = [
  {metric_names}
]

[output]
root = "artifacts"
save_vectors = {str(save_vectors).lower()}
save_plots = {str(save_plots).lower()}
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def test_canonical_baseline_config_is_complete_and_valid() -> None:
    config = reconstruction.load_config(Path("configs/reconstruction/baseline.toml"))

    assert config.name == "reconstruction-baseline"
    assert config.signal_names == tuple(reconstruction.SIGNAL_GENERATORS)
    assert config.metric_names == reconstruction.METRIC_NAMES
    assert config.repetitions == 5
    assert config.warmup_repetitions == 1
    assert config.save_vectors
    assert config.save_plots


def test_runner_persists_metrics_timings_vectors_environment_and_hashes(
    experiment_path: Path,
) -> None:
    config_path = _write_config(experiment_path / "config.toml")

    summary = reconstruction.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "benchmark"),
    )

    assert summary.n_conditions == 2
    assert summary.n_failures == 0
    assert summary.run_dir.parent == experiment_path.resolve() / "runs"
    expected_files = {
        "config.json",
        "environment.json",
        "manifest.json",
        "metrics.csv",
        "timings.csv",
    }
    assert expected_files <= {path.name for path in summary.run_dir.iterdir()}
    assert len(list((summary.run_dir / "vectors").glob("*.npz"))) == 2
    assert not (summary.run_dir / "plots").exists()

    with summary.metrics_path.open(encoding="utf-8", newline="") as stream:
        metrics_rows = list(csv.DictReader(stream))
    assert [row["status"] for row in metrics_rows] == ["ok", "ok"]
    assert [float(row["tolerance"]) for row in metrics_rows] == [0.01, 0.03]
    assert all(int(row["n_points"]) == 32 for row in metrics_rows)

    with (summary.run_dir / "timings.csv").open(encoding="utf-8", newline="") as stream:
        timing_rows = list(csv.DictReader(stream))
    assert len(timing_rows) == 8
    assert {row["phase"] for row in timing_rows} == {"transform", "reconstruction"}

    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 2
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "benchmark"]
    assert environment["config"]["resolved"]["signal_seeds"]["sine"] > 0
    assert isinstance(environment["git"]["dirty"], bool)

    manifest = json.loads((summary.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (summary.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_runner_renders_condition_and_summary_plots(experiment_path: Path) -> None:
    config_path = _write_config(
        experiment_path / "plot-config.toml",
        save_vectors=False,
        save_plots=True,
        repetitions=1,
        tolerances=(0.03,),
    )

    summary = reconstruction.run_experiment(config_path, output_root=experiment_path / "runs")

    plot_paths = sorted((summary.run_dir / "plots").glob("*.png"))
    assert len(plot_paths) == 4
    assert {path.name for path in plot_paths} == {
        "chain__sine__tol-0p03.png",
        "summary__compression-by-tolerance.png",
        "summary__compression-rmse-tradeoff.png",
        "summary__rmse-by-tolerance.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plot_paths)


def test_condition_failures_are_preserved_and_reported(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(experiment_path / "failure-config.toml", save_vectors=False)

    def fail_generator(**_: object) -> np.ndarray:
        raise RuntimeError("deliberate test failure")

    monkeypatch.setitem(reconstruction.SIGNAL_GENERATORS, "sine", fail_generator)
    summary = reconstruction.run_experiment(config_path, output_root=experiment_path / "runs")

    assert summary.n_conditions == 2
    assert summary.n_failures == 2
    with summary.metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["status"] for row in rows} == {"error"}
    assert {row["error_type"] for row in rows} == {"RuntimeError"}
    assert {row["error_message"] for row in rows} == {"deliberate test failure"}
    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete_with_failures"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("repetitions = 2", "repetitions = 0"),
        ('names = ["sine"]', 'names = ["sine", "sine"]'),
        ("tolerances = [0.01, 0.03]", "tolerances = [0.03, 0.01]"),
        ("frequency = 2.0", "unknown = 2.0"),
        ('"rmse"', '"not_a_metric"'),
    ],
)
def test_invalid_experiment_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_config(experiment_path / "invalid.toml")
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        reconstruction.load_config(config_path)
