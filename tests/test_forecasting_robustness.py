"""Tests for the factorial forecasting robustness runner."""

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

EXPERIMENTS_PATH = Path(__file__).parents[1] / "experiments"
FORECASTING_SPEC = importlib.util.spec_from_file_location(
    "forecasting", EXPERIMENTS_PATH / "forecasting.py"
)
if FORECASTING_SPEC is None or FORECASTING_SPEC.loader is None:
    raise RuntimeError("could not load forecasting experiment module")
if "forecasting" not in sys.modules:
    forecasting_module = importlib.util.module_from_spec(FORECASTING_SPEC)
    sys.modules[FORECASTING_SPEC.name] = forecasting_module
    FORECASTING_SPEC.loader.exec_module(forecasting_module)

ROBUSTNESS_SPEC = importlib.util.spec_from_file_location(
    "vectorchain_forecasting_robustness_runner",
    EXPERIMENTS_PATH / "forecasting_robustness.py",
)
if ROBUSTNESS_SPEC is None or ROBUSTNESS_SPEC.loader is None:
    raise RuntimeError("could not load forecasting robustness module")
robustness = importlib.util.module_from_spec(ROBUSTNESS_SPEC)
sys.modules[ROBUSTNESS_SPEC.name] = robustness
ROBUSTNESS_SPEC.loader.exec_module(robustness)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-forecast-robustness-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_base_config(path: Path) -> Path:
    content = """
[experiment]
name = "test-forecast-base"
seed = 11

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
save_models = true
save_plots = false
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _write_grid_config(path: Path, *, save_plots: bool = False) -> Path:
    content = f"""
[experiment]
name = "test-forecast-robustness"
base_config = "baseline.toml"
seeds = [11, 22]

[grid]
context_lengths = [8]
horizons = [1, 2]
tolerances = [0.01, 0.03]
stride = 4

[criteria]
predictive_parity_ratio = 1.10
maximum_step_fraction = 0.5
maximum_scalar_fraction = 1.0
robust_seed_rate = 0.5

[timing]
repetitions = 1
warmup_repetitions = 0

[output]
root = "artifacts"
save_plots = {str(save_plots).lower()}
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _write_configs(directory: Path, *, save_plots: bool = False) -> Path:
    _write_base_config(directory / "baseline.toml")
    return _write_grid_config(directory / "robustness.toml", save_plots=save_plots)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_canonical_robustness_config_registers_complete_grid() -> None:
    config = robustness.load_config(Path("configs/forecasting/robustness.toml"))

    assert config.name == "forecasting-robustness-grid"
    assert len(config.seeds) == 5
    assert config.context_lengths == (32, 64, 128)
    assert config.horizons == (1, 4, 16)
    assert config.tolerances == (0.01, 0.03, 0.1)
    assert config.robust_seed_rate == 0.8
    expected = len(config.seeds) * 3 * 3 * (2 + 3)
    assert expected == 225


def test_runner_persists_seed_level_and_aggregate_results(
    experiment_path: Path,
) -> None:
    config_path = _write_configs(experiment_path, save_plots=True)

    result = robustness.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "robustness"),
    )

    assert result.n_conditions == 16
    assert result.n_failures == 0
    assert {
        "conditions.csv",
        "conditions_by_signal.csv",
        "config.json",
        "environment.json",
        "manifest.json",
        "summary.csv",
        "summary_by_signal.csv",
    } <= {path.name for path in result.run_dir.iterdir()}

    conditions = _read_csv(result.run_dir / "conditions.csv")
    assert len(conditions) == 32
    assert {row["status"] for row in conditions} == {"ok"}
    assert {row["seed"] for row in conditions} == {"11", "22"}
    assert {row["horizon"] for row in conditions} == {"1", "2"}
    assert {row["context_length"] for row in conditions} == {"8"}
    vector_rows = [row for row in conditions if row["representation"] == "vectorchain"]
    assert {row["tolerance"] for row in vector_rows} == {"0.01", "0.03"}
    assert all(float(row["rmse_ratio_vs_raw"]) > 0.0 for row in vector_rows)
    assert all(row["joint_success"] in {"True", "False"} for row in vector_rows)

    assert len(_read_csv(result.run_dir / "conditions_by_signal.csv")) == 64
    summary = _read_csv(result.summary_path)
    assert len(summary) == 16
    assert {row["n_seeds"] for row in summary} == {"2"}
    vector_summary = [row for row in summary if row["representation"] == "vectorchain"]
    assert all(float(row["joint_success_rate"]) in {0.0, 0.5, 1.0} for row in vector_summary)
    assert all(row["robust_cell"] in {"True", "False"} for row in vector_summary)
    assert len(_read_csv(result.run_dir / "summary_by_signal.csv")) == 32

    plot_paths = sorted((result.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plot_paths} == {
        "summary__joint-success-rate.png",
        "summary__payload-parity-tradeoff.png",
        "summary__ratio-distributions.png",
        "summary__rmse-ratio.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plot_paths)

    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 16
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "robustness"]
    assert set(environment["derived_signal_seeds"]) == {"11", "22"}

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (result.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_condition_failures_are_counted_once_per_evaluation(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_configs(experiment_path)
    original = robustness.forecasting._run_representation

    def fail_raw(
        representation: str,
        examples: object,
        config: object,
    ) -> object:
        if representation == "raw":
            raise RuntimeError("deliberate robustness failure")
        return original(representation, examples, config)

    monkeypatch.setattr(robustness.forecasting, "_run_representation", fail_raw)
    result = robustness.run_experiment(config_path, output_root=experiment_path / "runs")

    assert result.n_conditions == 16
    assert result.n_failures == 4
    failed = [
        row for row in _read_csv(result.run_dir / "conditions.csv") if row["status"] == "error"
    ]
    assert len(failed) == 8
    assert {row["representation"] for row in failed} == {"raw"}
    assert {row["error_message"] for row in failed} == {"deliberate robustness failure"}
    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete_with_failures"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seeds = [11, 22]", "seeds = [11]"),
        ("context_lengths = [8]", "context_lengths = [9]"),
        ("horizons = [1, 2]", "horizons = [2, 1]"),
        ("tolerances = [0.01, 0.03]", "tolerances = [0.01, 0.02]"),
        ("maximum_step_fraction = 0.5", "maximum_step_fraction = 0.0"),
        ("robust_seed_rate = 0.5", "robust_seed_rate = 1.1"),
        ("repetitions = 1", "repetitions = 0"),
    ],
)
def test_invalid_robustness_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_configs(experiment_path)
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        robustness.load_config(config_path)
