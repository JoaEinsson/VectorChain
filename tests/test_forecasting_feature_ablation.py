"""Tests for the forecasting kinematic-feature ablation runner."""

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
ablation = _load_module(
    "vectorchain_forecasting_feature_ablation_runner",
    EXPERIMENTS_PATH / "forecasting_feature_ablation.py",
)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-feature-ablation-", dir=artifact_root))
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


def _write_ablation_config(path: Path, *, save_plots: bool = False) -> Path:
    content = f"""
[experiment]
name = "test-feature-ablation"
base_config = "baseline.toml"
seeds = [11, 22]

[grid]
context_lengths = [8]
horizons = [1, 2]
stride = 4
tolerance = 0.1

[ablations]
names = ["segment", "absolute_geometry", "turning_matched", "turning", "full_relational"]
reference_variant = "absolute_geometry"
primary_variant = "turning"
capacity_control_variant = "turning_matched"

[ablations.features]
segment = ["dt", "dy"]
absolute_geometry = ["dt", "dy", "theta", "r"]
turning_matched = ["dt", "dy", "theta", "delta_theta"]
turning = ["dt", "dy", "theta", "r", "delta_theta"]
full_relational = ["dt", "dy", "theta", "r", "delta_theta", "delta_r"]

[criteria]
primary_split = "validation"
maximum_primary_rmse_ratio = 0.99
maximum_capacity_control_rmse_ratio = 1.0
robust_seed_rate = 0.5
minimum_robust_cell_rate = 0.5
predictive_parity_ratio = 1.10
maximum_step_fraction = 0.5
maximum_scalar_fraction = 1.0

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
    return _write_ablation_config(directory / "feature_ablation.toml", save_plots=save_plots)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_canonical_config_registers_pre_specified_feature_grid() -> None:
    config = ablation.load_config(Path("configs/forecasting/feature_ablation.toml"))

    assert config.name == "forecasting-kinematic-feature-ablation"
    assert config.reference_variant == "absolute_geometry"
    assert config.primary_variant == "turning"
    assert config.capacity_control_variant == "turning_matched"
    assert config.primary_split == "validation"
    assert config.variant_features["turning"] == ("dt", "dy", "theta", "r", "delta_theta")
    expected = len(config.seeds) * 3 * 3 * (2 + len(config.variant_names))
    assert expected == 315


def test_runner_persists_auditable_ablation_and_gate(experiment_path: Path) -> None:
    config_path = _write_configs(experiment_path, save_plots=True)

    result = ablation.run_experiment(
        config_path,
        output_root=experiment_path / "runs",
        command_args=("pytest", "feature-ablation"),
    )

    assert result.n_conditions == 28
    assert result.n_failures == 0
    expected_files = {
        "conditions.csv",
        "conditions_by_signal.csv",
        "config.json",
        "environment.json",
        "gate.json",
        "manifest.json",
        "step_audit.csv",
        "summary.csv",
        "summary_by_seed.csv",
        "summary_by_signal.csv",
    }
    assert expected_files <= {path.name for path in result.run_dir.iterdir()}

    conditions = _read_csv(result.run_dir / "conditions.csv")
    assert len(conditions) == 56
    assert {row["status"] for row in conditions} == {"ok"}
    vector_rows = [row for row in conditions if row["representation"] == "vectorchain"]
    assert {row["variant"] for row in vector_rows} == {
        "segment",
        "absolute_geometry",
        "turning_matched",
        "turning",
        "full_relational",
    }
    parameter_counts = {
        row["variant"]: int(row["n_model_parameters"])
        for row in vector_rows
        if row["split"] == "validation"
    }
    assert parameter_counts["absolute_geometry"] == parameter_counts["turning_matched"]
    assert parameter_counts["turning"] > parameter_counts["absolute_geometry"]
    assert all(float(row["rmse_ratio_vs_reference"]) > 0.0 for row in vector_rows)

    assert len(_read_csv(result.run_dir / "conditions_by_signal.csv")) == 112
    assert len(_read_csv(result.summary_path)) == 28
    assert len(_read_csv(result.run_dir / "summary_by_seed.csv")) == 20
    assert len(_read_csv(result.run_dir / "summary_by_signal.csv")) == 56

    audits = _read_csv(result.run_dir / "step_audit.csv")
    assert len(audits) == 20
    signatures: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in audits:
        assert row["matches_reference"] == "True"
        signatures[(row["seed"], row["context_length"], row["horizon"])].add(
            row["step_signature_sha256"]
        )
    assert all(len(values) == 1 for values in signatures.values())

    gate = json.loads(result.gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "evaluated"
    assert isinstance(gate["passed"], bool)
    assert gate["primary_split"] == "validation"
    assert gate["observed"]["primary_cell_trials"] == 2

    plot_paths = sorted((result.run_dir / "plots").glob("*.png"))
    assert {path.name for path in plot_paths} == {
        "summary__payload-error-tradeoff.png",
        "summary__primary-effect-heatmap.png",
        "summary__seed-geometric-ratios.png",
        "summary__variant-ratio-distributions.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in plot_paths)

    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete"
    assert environment["n_conditions"] == 28
    assert environment["n_failures"] == 0
    assert environment["command"]["argv"] == ["pytest", "feature-ablation"]

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for entry in manifest["files"]:
        content = (result.run_dir / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_raw_failures_do_not_turn_a_scientific_result_into_an_execution_crash(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_configs(experiment_path)
    original = ablation.forecasting._run_representation

    def fail_raw(representation: str, examples: object, config: object) -> object:
        if representation == "raw":
            raise RuntimeError("deliberate ablation failure")
        return original(representation, examples, config)

    monkeypatch.setattr(ablation.forecasting, "_run_representation", fail_raw)
    result = ablation.run_experiment(config_path, output_root=experiment_path / "runs")

    assert result.n_conditions == 28
    assert result.n_failures == 4
    failed = [
        row for row in _read_csv(result.run_dir / "conditions.csv") if row["status"] == "error"
    ]
    assert len(failed) == 8
    assert {row["representation"] for row in failed} == {"raw"}
    environment = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "complete_with_failures"


def test_step_audit_rejects_feature_dependent_segmentation(experiment_path: Path) -> None:
    config = ablation.load_config(_write_configs(experiment_path))
    rows = {
        "absolute_geometry": [{"example_id": "sample", "input_steps": 3}],
        "turning": [{"example_id": "sample", "input_steps": 2}],
    }

    with pytest.raises(RuntimeError, match="input steps changed"):
        ablation._audit_steps(rows, config, seed=11, context_length=8, horizon=1)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seeds = [11, 22]", "seeds = [11]"),
        ("context_lengths = [8]", "context_lengths = [9]"),
        ('primary_split = "validation"', 'primary_split = "test"'),
        ("maximum_primary_rmse_ratio = 0.99", "maximum_primary_rmse_ratio = 1.0"),
        ("robust_seed_rate = 0.5", "robust_seed_rate = 0.0"),
        (
            'turning_matched = ["dt", "dy", "theta", "delta_theta"]',
            'turning_matched = ["dt", "dy", "delta_theta"]',
        ),
    ],
)
def test_invalid_ablation_configs_are_rejected(experiment_path: Path, old: str, new: str) -> None:
    config_path = _write_configs(experiment_path)
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(content.replace(old, new), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        ablation.load_config(config_path)
