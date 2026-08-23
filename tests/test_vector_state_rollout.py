"""Tests for the causal event-state forecasting and rollout runner."""

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
rollout = _load_module(
    "vectorchain_vector_state_rollout_runner",
    EXPERIMENTS_PATH / "vector_state_rollout.py",
)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-vector-state-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _write_base_config(path: Path) -> Path:
    content = """
[experiment]
name = "test-vector-state-base"
seed = 11

[signals]
names = ["sine", "ramp"]
n_points = 256
noise_std = 0.01

[signals.parameters.sine]
amplitude = 1.0
offset = 0.0
frequency = 4.0
phase = 0.0

[signals.parameters.ramp]
amplitude = 1.0
offset = 0.0

[forecast]
context_length = 8
horizon = 1
stride = 2

[split]
train_fraction = 0.6
validation_fraction = 0.2

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


def _write_rollout_config(path: Path, *, save_plots: bool = False) -> Path:
    content = f"""
[experiment]
name = "test-vector-state-rollout"
base_config = "baseline.toml"
phase = "test-stage10a"
seeds = [11, 22]

[signals]
n_points = 256

[event_state]
tolerance = 0.03
min_segment_length = 2
history_lengths = [2, 4]
candidate = "vectorchain_relational"
representations = ["vectorchain_cartesian", "vectorchain_absolute", "vectorchain_relational", "raw_matched", "fixed_relational", "raw_ar", "persistence"]

[event_state.features]
vectorchain_cartesian = ["dt", "dy", "open_dy"]
vectorchain_absolute = ["dt", "dy", "theta", "r", "open_dy"]
vectorchain_relational = ["dt", "dy", "theta", "r", "delta_theta", "delta_r", "open_dy"]
fixed_relational = ["dt", "dy", "theta", "r", "delta_theta", "delta_r", "open_dy"]

[targets]
names = ["log1p_remaining_dt", "remaining_dy", "next_open_dy"]
duration_projection = "round_expm1_clip"
maximum_remaining_dt = 32
zero_duration_displacement = "force_zero"

[controls]
gate_names = ["raw_matched", "fixed_relational", "raw_ar", "persistence"]
fixed_segment_length = "training_median_vectorchain_dt"
raw_lags_per_candidate_feature = 1

[split]
train_fraction = 0.6
validation_fraction = 0.2
minimum_examples_per_signal_split = 1

[model]
kind = "ridge"
alpha = 0.001

[rollout]
raw_horizons = [4, 8]
origin_event_stride = 2
maximum_predicted_events = 32

[criteria]
primary_split = "test"
primary_history_length = 2
maximum_candidate_rmse_ratio = 0.99
robust_seed_rate = 0.5
minimum_robust_horizon_rate = 0.5
required_postprojection_validity_rate = 1.0
required_rollout_completion_rate = 1.0
maximum_candidate_step_fraction_vs_raw_matched = 0.14285714285714285
maximum_candidate_scalar_fraction_vs_raw_matched = 1.0

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
    return _write_rollout_config(directory / "vector_state_rollout.toml", save_plots=save_plots)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_canonical_config_registers_unopened_stage10_grid() -> None:
    config = rollout.load_config(Path("configs/forecasting/vector_state_rollout.toml"))

    assert config.name == "forecasting-vector-state-rollout"
    assert config.phase == "stage10a-preregistered-synthetic"
    assert len(config.seeds) == 5
    assert config.tolerance == 0.03
    assert config.history_lengths == (4, 8, 16)
    assert config.primary_history_length == 8
    assert config.primary_split == "test"
    assert config.candidate == "vectorchain_relational"
    assert config.raw_horizons == (16, 64, 128)
    expected = 5 * 3 * 7 * 2 * 3
    assert expected == 630


def test_emission_state_is_prefix_invariant_and_excludes_finalize() -> None:
    time_axis = np.linspace(0.0, 8.0 * np.pi, 160)
    original = np.sin(time_axis) + 0.05 * np.sin(7.0 * time_axis)
    cut = 91
    modified = original.copy()
    modified[cut + 1 :] = np.linspace(20.0, -20.0, modified.size - cut - 1)

    left = rollout.extract_emission_states(original, tolerance=0.03, min_segment_length=2)
    right = rollout.extract_emission_states(modified, tolerance=0.03, min_segment_length=2)
    left_prefix = tuple(state for state in left if state.emitted_at <= cut)
    right_prefix = tuple(state for state in right if state.emitted_at <= cut)

    assert left_prefix
    assert left_prefix == right_prefix
    assert all(state.emitted_at == state.end + 1 for state in left)
    assert left[-1].end < original.size - 1
    assert all(state.dt >= 1 and np.isfinite(state.open_dy) for state in left)


def test_context_local_relations_do_not_read_an_unreported_state() -> None:
    values = np.sin(np.linspace(0.0, 10.0, 128))
    states = rollout.extract_emission_states(values, tolerance=0.01, min_segment_length=2)
    matrix = rollout._state_feature_matrix(
        states[-4:],
        ("dt", "dy", "theta", "r", "delta_theta", "delta_r", "open_dy"),
    )

    assert matrix.shape == (4, 7)
    assert matrix[0, 4] == 0.0
    assert matrix[0, 5] == 0.0


def test_multioutput_ridge_and_duration_projection_are_deterministic() -> None:
    inputs = np.arange(24, dtype=np.float64).reshape(8, 3)
    targets = np.column_stack((inputs[:, 0] * 0.1, inputs[:, 1] - inputs[:, 2]))
    left = rollout.fit_multi_ridge(inputs, targets, alpha=0.01)
    right = rollout.fit_multi_ridge(inputs, targets, alpha=0.01)

    assert np.array_equal(left.predict(inputs), right.predict(inputs))
    assert left.n_predictive_parameters == 8
    assert rollout.project_duration(np.log1p(2.5), 8).remaining_dt == 2
    negative = rollout.project_duration(-1.0, 8)
    assert negative.raw_invalid is True
    assert negative.remaining_dt == 0
    positive_infinity = rollout.project_duration(float("inf"), 8)
    assert positive_infinity.raw_invalid is True
    assert positive_infinity.remaining_dt == 8
    not_a_number = rollout.project_duration(float("nan"), 8)
    assert not_a_number.raw_invalid is True
    assert not_a_number.remaining_dt == 0


def test_runner_persists_causal_events_rollouts_and_gate(experiment_path: Path) -> None:
    config_path = _write_configs(experiment_path, save_plots=True)
    result = rollout.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "vector-state-rollout"),
    )

    assert result.n_conditions == 112
    assert result.n_failures == 0
    expected_files = {
        "conditions.csv",
        "conditions_by_signal.csv",
        "config.json",
        "environment.json",
        "event_predictions.csv",
        "events.csv",
        "gate.json",
        "manifest.json",
        "models.csv",
        "rollouts.csv",
        "summary.csv",
        "summary_by_seed.csv",
    }
    assert expected_files <= {path.name for path in result.run_dir.iterdir()}

    events = _read_csv(result.run_dir / "events.csv")
    assert events
    assert all(int(row["emitted_at"]) == int(row["end"]) + 1 for row in events)

    predictions = _read_csv(result.run_dir / "event_predictions.csv")
    assert predictions
    assert all(int(row["actual_remaining_dt"]) == int(row["actual_dt"]) - 1 for row in predictions)
    assert {row["postprojection_valid"] for row in predictions} == {"True"}

    models = _read_csv(result.run_dir / "models.csv")
    by_cell = {(row["seed"], row["history_length"], row["representation"]): row for row in models}
    for seed in ("11", "22"):
        for history in ("2", "4"):
            counts = {
                int(by_cell[(seed, history, name)]["n_predictive_parameters"])
                for name in (
                    "vectorchain_relational",
                    "raw_matched",
                    "fixed_relational",
                )
            }
            assert len(counts) == 1

    conditions = _read_csv(result.run_dir / "conditions.csv")
    assert len(conditions) == 112
    assert {row["status"] for row in conditions} == {"ok"}
    assert all(float(row["candidate_rmse_ratio_vs_control"]) > 0.0 for row in conditions)
    candidate = [row for row in conditions if row["representation"] == "vectorchain_relational"]
    matched = [row for row in conditions if row["representation"] == "raw_matched"]
    assert {float(row["mean_input_steps"]) for row in candidate} == {2.0, 4.0}
    assert {float(row["mean_input_steps"]) for row in matched} == {14.0, 28.0}
    assert all(float(row["rollout_completion_rate"]) == 1.0 for row in candidate)

    gate = json.loads(result.gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "evaluated"
    assert gate["hypothesis"] == "K5-A"
    assert gate["primary_split"] == "test"
    assert gate["primary_history_length"] == 2
    assert isinstance(gate["passed"], bool)

    plot_paths = sorted((result.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plot_paths} == {
        "summary__next-event-predictions.png",
        "summary__rollout-ratio-distributions.png",
        "summary__rollout-rmse-by-horizon.png",
        "summary__rollout-validity.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plot_paths)

    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 112
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "vector-state-rollout"]

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (result.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seeds = [11, 22]", "seeds = [11]"),
        ('primary_split = "test"', 'primary_split = "validation"'),
        ("primary_history_length = 2", "primary_history_length = 3"),
        ("maximum_candidate_rmse_ratio = 0.99", "maximum_candidate_rmse_ratio = 1.0"),
        ("raw_lags_per_candidate_feature = 1", "raw_lags_per_candidate_feature = 2"),
        (
            'vectorchain_cartesian = ["dt", "dy", "open_dy"]',
            'vectorchain_cartesian = ["dt", "dy"]',
        ),
    ],
)
def test_invalid_rollout_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_configs(experiment_path)
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        rollout.load_config(config_path)
