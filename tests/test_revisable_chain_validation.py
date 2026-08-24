"""Tests for the K7 train/validation-only runner and leakage barriers."""

import csv
import gzip
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS_PATH = Path(__file__).parents[1] / "experiments"


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "revisable_chain" not in sys.modules:
    _load_module("revisable_chain", EXPERIMENTS_PATH / "revisable_chain.py")
runner = _load_module(
    "vectorchain_revisable_chain_validation_runner",
    EXPERIMENTS_PATH / "revisable_chain_validation.py",
)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-revisable-validation-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_config(path: Path, *, seed: int = 11) -> Path:
    path.write_text(
        f"""
[experiment]
name = "test-revisable-chain-validation"
phase = "test-stage12a"
scope = "fixture"
seeds = [{seed}]

[signals]
names = ["baseline_modulation"]
n_points = 512
noise_std = 0.02

[split]
train_fraction = 0.5
validation_fraction = 0.2
inner_train_fraction = 0.8

[selection]
lambda_revision = [0.01, 0.1, 1.0]
lambda_bend = [0.01, 0.1, 1.0]
representation = "revisable_absolute"

[model]
kind = "ridge"
alpha = 0.001

[output]
root = "artifacts"
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_development_config_is_exact_and_cannot_authorize_test() -> None:
    config = runner.load_config(Path("configs/forecasting/revisable_chain_development.toml"))

    assert config.scope == "development"
    assert config.seeds == (11, 22)
    assert config.mechanisms == (
        "frequency_modulation",
        "baseline_modulation",
        "crest_asymmetry_modulation",
    )
    assert config.lambda_revision == (0.01, 0.1, 1.0)
    assert config.lambda_bend == (0.01, 0.1, 1.0)


def test_fixture_scope_rejects_every_seed_outside_development(experiment_path: Path) -> None:
    config_path = _write_config(experiment_path / "config.toml", seed=23)

    with pytest.raises(ValueError, match="development seeds"):
        runner.load_config(config_path)


def test_endpoint_mask_excludes_rows_crossing_split_boundaries() -> None:
    indices = np.asarray(
        (
            (10, 17, 41),
            (40, 47, 71),
            (50, 57, 81),
            (70, 77, 101),
        ),
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        runner.endpoint_mask(indices, start_inclusive=0, end_exclusive=50),
        (True, False, False, False),
    )
    np.testing.assert_array_equal(
        runner.endpoint_mask(indices, start_inclusive=50, end_exclusive=100),
        (False, False, True, False),
    )


def test_multioutput_ridge_is_deterministic_and_has_registered_capacity() -> None:
    inputs = np.arange(40 * 17, dtype=np.float64).reshape(40, 17)
    inputs[:, 3] = 1.0
    targets = np.column_stack((inputs[:, 0], inputs[:, 1] - inputs[:, 2], inputs[:, 4]))

    left = runner.fit_multi_ridge(inputs, targets, alpha=0.001)
    right = runner.fit_multi_ridge(inputs, targets, alpha=0.001)

    assert left.n_predictive_parameters == 54
    np.testing.assert_array_equal(left.scale_, right.scale_)
    np.testing.assert_array_equal(left.coefficients_, right.coefficients_)
    np.testing.assert_array_equal(left.predict(inputs), right.predict(inputs))
    assert left.scale_[3] == 1.0


def test_runner_selects_and_validates_without_materializing_test(
    experiment_path: Path,
) -> None:
    config_path = _write_config(experiment_path / "config.toml")
    output_root = experiment_path / "outputs"

    result = runner.run_validation(
        config_path,
        output_root=output_root,
        command_args=("pytest", "revisable-chain-validation"),
    )

    expected_files = {
        "causality_audit.csv",
        "commit_audit.csv",
        "config.json",
        "environment.json",
        "failures.csv",
        "inputs.csv.gz",
        "manifest.json",
        "metrics.csv",
        "models.npz",
        "origins.csv",
        "predictions.csv",
        "selection.json",
        "selection_metrics.csv",
        "solver_audit.csv",
        "working_state.csv.gz",
    }
    assert {path.name for path in result.run_dir.iterdir()} == expected_files
    assert not (result.run_dir / "gate.json").exists()

    selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
    assert selection["status"] == "fixture_train_only"
    assert selection["decisory"] is False
    assert selection["test_materialized"] is False
    assert len(selection["grid"]) == 9
    assert len(_read_csv(result.run_dir / "selection_metrics.csv")) == 27

    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["decisory"] is False
    assert environment["test_materialized"] is False
    assert environment["config"]["resolved"]["generated_stop_exclusive"] == 358
    assert environment["command"]["argv"] == ["pytest", "revisable-chain-validation"]

    metrics = _read_csv(result.metrics_path)
    assert len(metrics) == 18
    assert {row["split"] for row in metrics} == {"validation"}
    assert {row["representation"] for row in metrics} == {
        "immutable_absolute",
        "revisable_absolute",
        "revisable_spatial",
        "revisable_temporal",
        "raw_matched",
        "persistence",
    }
    trained_metrics = [row for row in metrics if row["representation"] != "persistence"]
    assert all(row["input_scalars"] == "17" for row in trained_metrics)
    assert all(row["input_bytes"] == "136" for row in trained_metrics)
    assert all(row["n_predictive_parameters"] == "54" for row in trained_metrics)
    assert all(int(row["input_rank"]) <= 17 for row in trained_metrics)
    assert all(int(row["model_state_bytes"]) > 0 for row in trained_metrics)
    origins = _read_csv(result.run_dir / "origins.csv")
    assert max(int(row["target_32"]) for row in origins) < 358
    assert all(row["split"] != "test" for row in origins)
    assert all(row["passed"] == "1" for row in _read_csv(result.run_dir / "causality_audit.csv"))
    assert all(
        row["structural_pass"] == "1" for row in _read_csv(result.run_dir / "solver_audit.csv")
    )
    assert _read_csv(result.run_dir / "failures.csv") == []

    with gzip.open(result.run_dir / "inputs.csv.gz", "rt", encoding="utf-8") as stream:
        assert sum(1 for _ in stream) > 1
    with np.load(result.run_dir / "models.npz") as models:
        assert len(models.files) == 20

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert "manifest.json" not in {item["path"] for item in manifest["files"]}
