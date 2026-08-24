"""Tests for the lock-bound K7 test runner without opening canonical signals."""

import importlib.util
import json
import shutil
import sys
import tempfile
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


if "revisable_chain" not in sys.modules:
    _load_module("revisable_chain", EXPERIMENTS_PATH / "revisable_chain.py")
if "revisable_chain_validation" not in sys.modules:
    _load_module("revisable_chain_validation", EXPERIMENTS_PATH / "revisable_chain_validation.py")
runner = _load_module(
    "vectorchain_revisable_chain_test_runner",
    EXPERIMENTS_PATH / "revisable_chain_test.py",
)


@pytest.fixture
def experiment_path() -> Iterator[Path]:
    artifact_root = Path("artifacts")
    artifact_root.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pytest-revisable-test-", dir=artifact_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def test_canonical_config_consumes_exact_frozen_selection_without_signal_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_signal_generation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configuration loading must not materialize any signal")

    monkeypatch.setattr(runner.revisable_chain, "generate_k7_signal", forbid_signal_generation)
    config = runner.load_config(Path("configs/forecasting/revisable_chain_test.toml"))

    assert config.scope == "canonical_test"
    assert config.selection_lock_sha256 == runner.EXPECTED_SELECTION_LOCK_SHA256
    assert config.selection["selected"] == {
        "global_nrmse": 0.5230412893176433,
        "lambda_bend": 1.0,
        "lambda_revision": 0.1,
    }
    assert config.selection_config.seeds == runner.validation.CANONICAL_SELECTION_SEEDS
    assert config.save_plots is True


def test_dirty_worktree_is_rejected_before_the_test_is_opened(
    experiment_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.validation, "_git_state", lambda _root: ("dirty-commit", True))

    def forbid_signal_generation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dirty refusal must precede full-signal generation")

    monkeypatch.setattr(runner.revisable_chain, "generate_k7_signal", forbid_signal_generation)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        runner.run_test(
            Path("configs/forecasting/revisable_chain_test.toml"),
            output_root=experiment_path,
            command_args=("pytest", "canonical-test-dirty"),
        )

    run_dirs = tuple((experiment_path / "revisable-chain-stage12a-test").iterdir())
    assert len(run_dirs) == 1
    environment = json.loads((run_dirs[0] / "environment.json").read_text(encoding="utf-8"))
    assert environment["status"] == "failed"
    assert environment["test_opened"] is False
    assert environment["test_materialized"] is False
    assert not (run_dirs[0] / "gate.json").exists()


def test_gate_passes_only_when_all_predictive_energy_and_structural_subgates_pass() -> None:
    metrics = _metric_rows(
        immutable=1.0,
        revisable_absolute=0.98,
        revisable_spatial=0.98,
        revisable_temporal=0.96,
        raw_matched=0.95,
    )
    structural = _structural_rows()

    gate, rows = runner.evaluate_gate(metrics, *structural, test_start=2867)

    assert gate["passed"] is True
    assert gate["subgates"]["K7-R"]["passed"] is True
    assert gate["subgates"]["K7-D"]["passed"] is True
    assert gate["subgates"]["K7-U"]["passed"] is True
    assert gate["subgates"]["structural"]["passed"] is True
    assert len(rows["comparison_cells"]) == 4 * 3 * 5 * 3
    assert len(rows["bootstrap_summary"]) == 4
    assert all(row["n_series"] == 15 for row in rows["bootstrap_summary"])


def test_negative_temporal_comparison_is_preserved_as_scientific_failure() -> None:
    metrics = _metric_rows(
        immutable=1.0,
        revisable_absolute=0.98,
        revisable_spatial=0.98,
        revisable_temporal=1.10,
        raw_matched=1.10,
    )
    structural = _structural_rows()

    gate, rows = runner.evaluate_gate(metrics, *structural, test_start=2867)

    assert gate["passed"] is False
    assert gate["subgates"]["K7-R"]["passed"] is True
    assert gate["subgates"]["K7-D"]["passed"] is False
    failed_cells = [
        row
        for row in rows["comparison_cells"]
        if row["comparison"] == "K7-D-absolute" and row["passed"] == 0
    ]
    assert len(failed_cells) == 45


def test_second_primary_is_blocked_but_explicit_replication_is_authorized(
    experiment_path: Path,
) -> None:
    experiment_root = experiment_path / "revisable-chain-stage12a-test"
    primary = experiment_root / "primary"
    primary.mkdir(parents=True)
    (primary / "environment.json").write_text(
        json.dumps({"mode": "primary", "test_opened": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="already opened"):
        runner._validate_run_authority(experiment_root, mode="primary", primary_dir=None)

    runner._validate_run_authority(experiment_root, mode="replication", primary_dir=primary)


def test_fixture_fit_uses_train_only_and_materializes_only_development_seed(
    experiment_path: Path,
) -> None:
    selection_config = runner.validation.load_config(
        _write_fixture_selection_config(experiment_path / "fixture-selection.toml")
    )

    models, rows = runner._fit_and_test(
        selection_config,
        selected=(0.1, 1.0),
        train_end=256,
        test_start=358,
    )

    assert len(models) == 5
    assert len(rows["metrics"]) == 18
    assert {row["split"] for row in rows["metrics"]} == {"test"}
    assert {row["seed"] for row in rows["metrics"]} == {11}
    assert len(rows["batch_stream_audit"]) == 1
    assert rows["batch_stream_audit"][0]["passed"] == 1
    assert all(row["structural_pass"] == 1 for row in rows["solver_audit"])
    assert all("raw_start_value" in row and "raw_end_value" in row for row in rows["working_state"])


def test_primary_config_comparison_is_semantic_across_windows_newlines(
    experiment_path: Path,
) -> None:
    config = runner.load_config(Path("configs/forecasting/revisable_chain_test.toml"))
    git_commit, _dirty = runner.validation._git_state(Path.cwd())
    primary = experiment_path / "primary"
    primary.mkdir()
    (primary / "environment.json").write_text(
        json.dumps(
            {
                "mode": "primary",
                "status": "complete",
                "git": {"commit": git_commit},
                "selection_lock_sha256": config.selection_lock_sha256,
            }
        ),
        encoding="utf-8",
    )
    expected = runner._resolved_config(config, *runner._split_bounds(config.selection_config))
    windows_json = json.dumps(expected, indent=2, sort_keys=True).replace("\n", "\r\n") + "\r\n"
    (primary / "config.json").write_bytes(windows_json.encode("utf-8"))

    runner._validate_primary(primary, git_commit, config)


def _write_fixture_selection_config(path: Path) -> Path:
    path.write_text(
        """
[experiment]
name = "fixture-revisable-chain-selection"
phase = "fixture-stage12a"
scope = "fixture"
seeds = [11]

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


def _metric_rows(
    *,
    immutable: float,
    revisable_absolute: float,
    revisable_spatial: float,
    revisable_temporal: float,
    raw_matched: float,
) -> list[dict[str, object]]:
    values = {
        "immutable_absolute": immutable,
        "revisable_absolute": revisable_absolute,
        "revisable_spatial": revisable_spatial,
        "revisable_temporal": revisable_temporal,
        "raw_matched": raw_matched,
        "persistence": 1.2,
    }
    return [
        {
            "mechanism": mechanism,
            "seed": seed,
            "horizon": horizon,
            "representation": representation,
            "rmse": rmse,
        }
        for mechanism in runner.revisable_chain.MECHANISM_NAMES
        for seed in runner.validation.CANONICAL_SELECTION_SEEDS
        for horizon in runner.revisable_chain.HORIZONS
        for representation, rmse in values.items()
    ]


def _structural_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    working_state: list[dict[str, object]] = []
    solver: list[dict[str, object]] = []
    causality: list[dict[str, object]] = []
    batch_stream: list[dict[str, object]] = []
    committed: list[dict[str, object]] = []
    for mechanism in runner.revisable_chain.MECHANISM_NAMES:
        for seed in runner.validation.CANONICAL_SELECTION_SEEDS:
            base = {"mechanism": mechanism, "seed": seed}
            working_state.extend(
                (
                    {
                        **base,
                        "observed_at": 3000,
                        "latent_region": "changing",
                        "correction_energy": 0.2,
                        "dt": 2,
                        "start_value": 0.0,
                        "end_value": 1.0,
                        "dy": 1.0,
                        "theta": 0.1,
                        "r": 2.1,
                        "update_theta": 0.1,
                        "update_r": 0.1,
                    },
                    {
                        **base,
                        "observed_at": 3001,
                        "latent_region": "stationary",
                        "correction_energy": 0.1,
                        "dt": 2,
                        "start_value": 0.0,
                        "end_value": 1.0,
                        "dy": 1.0,
                        "theta": 0.1,
                        "r": 2.1,
                        "update_theta": 0.1,
                        "update_r": 0.1,
                    },
                )
            )
            solver.append(
                {
                    **base,
                    "structural_pass": 1,
                    "n_links": 4,
                    "raw_span": 32,
                    "start_anchor_error": 0.0,
                    "current_anchor_error": 0.0,
                }
            )
            causality.append({**base, "passed": 1})
            batch_stream.append({**base, "passed": 1})
            committed.append({**base, "immutable_snapshot": 1})
    return working_state, solver, causality, batch_stream, committed
