"""Tests for the Stage-8 paired forecasting controls."""

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

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


if "forecasting" not in sys.modules:
    _load_module("forecasting", EXPERIMENTS_PATH / "forecasting.py")
if "forecasting_robustness" not in sys.modules:
    _load_module("forecasting_robustness", EXPERIMENTS_PATH / "forecasting_robustness.py")
controls = _load_module(
    "vectorchain_forecasting_controls_runner", EXPERIMENTS_PATH / "forecasting_controls.py"
)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-forecast-controls-", dir=artifact_root))
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
repetitions = 1
warmup_repetitions = 0

[output]
root = "artifacts"
save_models = false
save_plots = false
plot_dpi = 80
""".lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _write_control_config(path: Path, *, save_plots: bool = False) -> Path:
    content = f"""
[experiment]
name = "test-forecast-controls"
base_config = "baseline.toml"
seeds = [11, 22]

[grid]
context_lengths = [8]
horizons = [1, 2]
stride = 4
tolerance = 0.1

[candidate]
name = "absolute_geometry"
features = ["dt", "dy", "theta", "r"]

[controls]
names = ["local_geometry", "moving_average_geometry", "ewma_geometry", "fixed_geometry"]
gate_names = ["local_geometry", "moving_average_geometry", "ewma_geometry", "fixed_geometry"]
primary_name = "local_geometry"

[controls.parameters]
inner_train_fraction = 0.8
moving_average_windows = [2, 4, 8]
ewma_alphas = [0.2, 0.5, 0.8]
fixed_segment_lengths = [2, 4, 8]
abba_tolerances = [0.03, 0.1, 0.3]

[external]
fabba_version = "1.5.2"

[criteria]
primary_split = "validation"
maximum_candidate_rmse_ratio = 0.99
pareto_rmse_ratio = 1.01
robust_seed_rate = 0.5
minimum_robust_cell_rate = 0.5

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
    return _write_control_config(directory / "controls.toml", save_plots=save_plots)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_canonical_config_freezes_stage_8_grid_and_external_version() -> None:
    config = controls.load_config(Path("configs/forecasting/controls.toml"))

    assert config.name == "forecasting-absolute-geometry-controls"
    assert config.candidate_features == ("dt", "dy", "theta", "r")
    assert config.primary_name == "local_geometry"
    assert "abba_geometry" in config.control_names
    assert "abba_geometry" not in config.gate_names
    assert config.fabba_version == "1.5.2"
    expected = len(config.seeds) * 3 * 3 * (3 + len(config.control_names))
    assert expected == 360


def test_local_and_smoothing_geometry_are_prefix_causal() -> None:
    prefix = np.asarray([0.0, 1.0, 0.5, 2.0, 1.5])
    extended = np.concatenate((prefix, np.asarray([100.0, -200.0])))
    features = ("dt", "dy", "theta", "r")

    np.testing.assert_allclose(
        controls._local_geometry(prefix, features),
        controls._local_geometry(extended[: prefix.size], features),
    )
    np.testing.assert_allclose(
        controls._trailing_mean(prefix, 3), controls._trailing_mean(extended, 3)[: prefix.size]
    )
    np.testing.assert_allclose(controls._ewma(prefix, 0.2), controls._ewma(extended, 0.2)[:5])


def test_smoothing_formulas_are_trailing_and_recursive() -> None:
    values = np.asarray([1.0, 3.0, 5.0, 7.0])

    np.testing.assert_allclose(controls._trailing_mean(values, 3), [1.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(controls._ewma(values, 0.5), [1.0, 2.0, 3.5, 5.25])


def test_abba_adapter_uses_only_duration_and_increment(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(compress=lambda values, tol: [[2.0, 1.0, 999.0], [3.0, -2.0, 888.0]])
    monkeypatch.setattr(controls, "import_module", lambda name: fake)

    sequence = controls._abba_geometry(np.asarray([0.0, 1.0, 0.0]), 0.1, ("dt", "dy", "theta", "r"))

    assert sequence.shape == (2, 4)
    np.testing.assert_array_equal(sequence[:, :2], [[2.0, 1.0], [3.0, -2.0]])


def test_runner_persists_tuning_capacity_and_gate(experiment_path: Path) -> None:
    config_path = _write_configs(experiment_path, save_plots=True)

    result = controls.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "forecast-controls"),
    )

    assert result.n_conditions == 28
    assert result.n_failures == 0
    assert {
        "conditions.csv",
        "conditions_by_signal.csv",
        "config.json",
        "environment.json",
        "gate.json",
        "manifest.json",
        "summary.csv",
        "summary_by_seed.csv",
        "summary_by_signal.csv",
        "tuning.csv",
    } <= {path.name for path in result.run_dir.iterdir()}

    conditions = _read_csv(result.run_dir / "conditions.csv")
    assert len(conditions) == 56
    assert {row["status"] for row in conditions} == {"ok"}
    geometric = [
        row
        for row in conditions
        if row["representation"]
        in {
            "absolute_geometry",
            "local_geometry",
            "moving_average_geometry",
            "ewma_geometry",
            "fixed_geometry",
        }
    ]
    assert {row["n_pooled_features"] for row in geometric} == {"12"}
    assert {row["n_model_parameters"] for row in geometric} == {"13"}
    control_rows = [row for row in geometric if row["role"] != "candidate"]
    assert all(float(row["candidate_rmse_ratio_vs_control"]) > 0.0 for row in control_rows)
    assert all(row["candidate_pareto_vs_control"] in {"True", "False"} for row in control_rows)

    tuning = _read_csv(result.run_dir / "tuning.csv")
    assert len(tuning) == 36
    assert sum(row["selected"] == "True" for row in tuning) == 12
    assert {row["status"] for row in tuning} == {"ok"}
    assert all(int(row["n_inner_train"]) > 0 for row in tuning)
    assert all(int(row["n_inner_validation"]) > 0 for row in tuning)

    assert len(_read_csv(result.run_dir / "conditions_by_signal.csv")) == 112
    assert len(_read_csv(result.summary_path)) == 28
    assert len(_read_csv(result.run_dir / "summary_by_seed.csv")) == 16
    assert len(_read_csv(result.run_dir / "summary_by_signal.csv")) == 56

    gate = json.loads(result.gate_path.read_text(encoding="utf-8"))
    assert isinstance(gate["passed"], bool)
    assert gate["execution_checks"] == {
        "complete_condition_grid": True,
        "matched_downstream_capacity": True,
        "zero_condition_failures": True,
    }
    assert set(gate["controls"]) == {
        "local_geometry",
        "moving_average_geometry",
        "ewma_geometry",
        "fixed_geometry",
    }

    plots = sorted((result.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plots} == {
        "summary__control-ratio-distributions.png",
        "summary__gate-cell-coverage.png",
        "summary__primary-control-heatmap.png",
        "summary__tuning-selections.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plots)

    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["command"]["argv"] == ["pytest", "forecast-controls"]
    assert environment["experimental_dependencies"] == {}

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        content = (result.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_tuning_score_ignores_outer_validation_and_test_contexts(experiment_path: Path) -> None:
    config = controls.load_config(_write_configs(experiment_path))
    condition = controls._condition_config(config, 11, 8, 1)
    signals, signal_seeds = controls.forecasting._generate_signals(condition)
    examples = controls.forecasting._build_examples(signals, signal_seeds, condition)
    transform = controls._control_transform("moving_average_geometry", 2, config)
    original = controls._inner_validation_score(examples, transform, condition, 0.8)
    altered = tuple(
        example
        if example.split == "train"
        else controls.replace(example, context=np.full_like(example.context, 1e9))
        for example in examples
    )

    repeated = controls._inner_validation_score(altered, transform, condition, 0.8)

    assert repeated == original


def test_condition_failures_are_preserved_and_fail_execution_gate(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_configs(experiment_path)
    original = controls.forecasting._run_representation

    def fail_raw(
        representation: str,
        examples: object,
        config: object,
        *,
        transform: object = None,
    ) -> object:
        if representation == "raw":
            raise RuntimeError("deliberate control failure")
        return original(representation, examples, config, transform=transform)

    monkeypatch.setattr(controls.forecasting, "_run_representation", fail_raw)
    result = controls.run_experiment(config_path, output_root=experiment_path / "runs")

    assert result.n_conditions == 28
    assert result.n_failures == 4
    assert result.gate_passed is False
    failed = [
        row for row in _read_csv(result.run_dir / "conditions.csv") if row["status"] == "error"
    ]
    assert len(failed) == 8
    assert {row["representation"] for row in failed} == {"raw"}
    gate = json.loads(result.gate_path.read_text(encoding="utf-8"))
    assert gate["execution_checks"]["zero_condition_failures"] is False


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seeds = [11, 22]", "seeds = [11]"),
        ("context_lengths = [8]", "context_lengths = [9]"),
        ('features = ["dt", "dy", "theta", "r"]', 'features = ["dt", "dy"]'),
        ('primary_split = "validation"', 'primary_split = "test"'),
        ("moving_average_windows = [2, 4, 8]", "moving_average_windows = [2, 4]"),
        ("inner_train_fraction = 0.8", "inner_train_fraction = 1.0"),
        ("maximum_candidate_rmse_ratio = 0.99", "maximum_candidate_rmse_ratio = 1.0"),
        (
            'gate_names = ["local_geometry", "moving_average_geometry", "ewma_geometry", "fixed_geometry"]',
            'gate_names = ["abba_geometry"]',
        ),
    ],
)
def test_invalid_control_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_configs(experiment_path)
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        controls.load_config(config_path)
