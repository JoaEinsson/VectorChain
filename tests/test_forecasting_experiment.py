"""Tests for the leakage-safe minimal forecasting runner."""

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

RUNNER_PATH = Path(__file__).parents[1] / "experiments" / "forecasting.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("vectorchain_forecasting_runner", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError("could not load forecasting experiment module")
forecasting = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = forecasting
RUNNER_SPEC.loader.exec_module(forecasting)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-forecasting-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_config(
    path: Path,
    *,
    save_models: bool = True,
    save_plots: bool = False,
) -> Path:
    content = f"""
[experiment]
name = "test-forecasting"
seed = 31415

[signals]
names = ["sine", "ramp"]
n_points = 64
noise_std = 0.01

[signals.parameters.sine]
amplitude = 1.0
offset = 0.0
frequency = 2.0
phase = 0.0

[signals.parameters.ramp]
amplitude = 1.0
offset = 0.0

[forecast]
context_length = 8
horizon = 1
stride = 2

[split]
train_fraction = 0.5
validation_fraction = 0.25

[representations]
names = ["raw", "first_difference", "vectorchain"]
summary_statistics = ["last", "mean", "std"]

[vectorchain]
causal = true
tolerance = 0.03
min_segment_length = 2
features = ["dt", "dy"]

[model]
kind = "ridge"
alpha = 0.001
repetitions = 2
warmup_repetitions = 0

[output]
root = "artifacts"
save_models = {str(save_models).lower()}
save_plots = {str(save_plots).lower()}
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_sequence_summary_uses_registered_columnwise_order() -> None:
    sequence = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    summary = forecasting.summarize_sequence(sequence)

    assert summary == pytest.approx([3.0, 4.0, 2.0, 3.0, 1.0, 1.0])


def test_ridge_fits_training_scaler_and_predicts_without_mutation() -> None:
    inputs = np.arange(5.0)[:, np.newaxis]
    targets = 2.0 * inputs[:, 0] + 1.0

    model = forecasting.fit_ridge(inputs, targets, alpha=1e-9)
    first = model.predict(inputs)
    second = model.predict(inputs)

    assert model.mean_ == pytest.approx([2.0])
    assert model.scale_ == pytest.approx([np.sqrt(2.0)])
    assert first == pytest.approx(targets, abs=1e-8)
    assert np.array_equal(first, second)
    assert not model.mean_.flags.writeable
    assert not model.coefficients_.flags.writeable
    assert model.state_bytes == 32


@pytest.mark.parametrize(
    ("inputs", "targets", "alpha", "error"),
    [
        (np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), 0.1, ValueError),
        (np.ones((2, 1)), np.asarray([1.0]), 0.1, ValueError),
        (np.ones((2, 1)), np.asarray([1.0, np.nan]), 0.1, ValueError),
        (np.ones((2, 1)), np.asarray([1.0, 2.0]), 0.0, ValueError),
    ],
)
def test_invalid_ridge_inputs_are_rejected(
    inputs: np.ndarray, targets: np.ndarray, alpha: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        forecasting.fit_ridge(inputs, targets, alpha=alpha)


def test_canonical_forecasting_config_is_complete_and_valid() -> None:
    config = forecasting.load_config(Path("configs/forecasting/baseline.toml"))

    assert config.name == "minimal-forecasting-baseline"
    assert config.signal_names == tuple(forecasting.SIGNAL_GENERATORS)
    assert config.representation_names == forecasting.REPRESENTATION_NAMES
    assert config.summary_statistics == forecasting.SUMMARY_STATISTICS
    assert config.context_length == 64
    assert config.horizon == 1
    assert config.vectorchain_causal
    assert config.save_models
    assert config.save_plots


def test_runner_preserves_temporal_split_and_complete_audit_trail(
    experiment_path: Path,
) -> None:
    config_path = _write_config(experiment_path / "config.toml", save_plots=True)

    summary = forecasting.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "forecasting"),
    )

    assert summary.n_conditions == 3
    assert summary.n_failures == 0
    assert {
        "config.json",
        "environment.json",
        "examples.csv",
        "inputs.csv",
        "manifest.json",
        "metrics.csv",
        "metrics_by_signal.csv",
        "models.npz",
        "naive_metrics.csv",
        "predictions.csv",
        "timings.csv",
    } <= {path.name for path in summary.run_dir.iterdir()}

    examples = _read_csv(summary.run_dir / "examples.csv")
    assert len(examples) == 56
    assert {row["split"] for row in examples} == {"train", "validation", "test"}
    for row in examples:
        origin = int(row["origin"])
        target = int(row["target_index"])
        assert int(row["context_start"]) == origin - 7
        assert target == origin + 1
        if row["split"] == "train":
            assert target < 32
        elif row["split"] == "validation":
            assert 32 <= target < 48
        else:
            assert target >= 48

    inputs = _read_csv(summary.run_dir / "inputs.csv")
    assert len(inputs) == 168
    ids_by_representation: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        ids_by_representation[row["representation"]].add(row["example_id"])
    assert len(ids_by_representation) == 3
    assert len({frozenset(ids) for ids in ids_by_representation.values()}) == 1

    metrics = _read_csv(summary.metrics_path)
    assert len(metrics) == 6
    assert {row["status"] for row in metrics} == {"ok"}
    assert {row["split"] for row in metrics} == {"validation", "test"}
    assert {row["representation"] for row in metrics} == set(forecasting.REPRESENTATION_NAMES)
    vectorchain_test = next(
        row for row in metrics if row["representation"] == "vectorchain" and row["split"] == "test"
    )
    assert vectorchain_test["predictive_parity_vs_raw"] in {"True", "False"}
    assert vectorchain_test["joint_success"] in {"True", "False"}
    assert float(vectorchain_test["step_reduction_factor_vs_raw"]) > 0.0

    assert len(_read_csv(summary.run_dir / "metrics_by_signal.csv")) == 12
    assert len(_read_csv(summary.run_dir / "naive_metrics.csv")) == 6
    assert len(_read_csv(summary.run_dir / "predictions.csv")) == 96
    timings = _read_csv(summary.run_dir / "timings.csv")
    assert len(timings) == 18
    assert {row["phase"] for row in timings} == {"train", "inference"}

    with np.load(summary.run_dir / "models.npz") as models:
        assert len(models.files) == 12
        assert "raw__coefficients" in models.files
        assert "vectorchain__scale" in models.files

    plots = sorted((summary.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plots} == {
        "summary__error-payload-tradeoff.png",
        "summary__input-size.png",
        "summary__test-error.png",
        "summary__test-rmse-by-signal.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plots)

    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 3
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "forecasting"]
    assert environment["config"]["resolved"]["train_end_exclusive"] == 32
    assert environment["config"]["resolved"]["validation_end_exclusive"] == 48

    manifest = json.loads((summary.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (summary.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_representation_failure_is_preserved(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(experiment_path / "failure.toml", save_models=False)
    original_transform = forecasting._transform

    def fail_raw(representation: str, context: np.ndarray, config: object) -> np.ndarray:
        if representation == "raw":
            raise RuntimeError("deliberate forecasting failure")
        return original_transform(representation, context, config)

    monkeypatch.setattr(forecasting, "_transform", fail_raw)
    summary = forecasting.run_experiment(config_path, output_root=experiment_path / "runs")

    assert summary.n_conditions == 3
    assert summary.n_failures == 1
    failed = [row for row in _read_csv(summary.metrics_path) if row["status"] == "error"]
    assert len(failed) == 2
    assert {row["representation"] for row in failed} == {"raw"}
    assert {row["error_type"] for row in failed} == {"RuntimeError"}
    assert {row["error_message"] for row in failed} == {"deliberate forecasting failure"}
    environment = json.loads((summary.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete_with_failures"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("context_length = 8", "context_length = 64"),
        ("train_fraction = 0.5", "train_fraction = 0.0"),
        ("validation_fraction = 0.25", "validation_fraction = 0.6"),
        (
            'names = ["raw", "first_difference", "vectorchain"]',
            'names = ["raw", "vectorchain"]',
        ),
        ('summary_statistics = ["last", "mean", "std"]', 'summary_statistics = ["last"]'),
        ("causal = true", "causal = false"),
        ("alpha = 0.001", "alpha = 0.0"),
    ],
)
def test_invalid_forecasting_configs_are_rejected(
    experiment_path: Path, old: str, new: str
) -> None:
    config_path = _write_config(experiment_path / "invalid.toml")
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        forecasting.load_config(config_path)
