"""Tests for the complete revision-path eliminatory representation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

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
    "vectorchain_revision_path_kill_test", EXPERIMENTS_PATH / "revision_path_kill_test.py"
)
revisable_chain = sys.modules["revisable_chain"]


def _bundle(*, stop_exclusive: int | None = None) -> object:
    signal = revisable_chain.generate_k7_signal(
        "baseline_modulation",
        seed=1103,
        n_points=512,
        stop_exclusive=stop_exclusive,
    )
    return revisable_chain.build_k7_designs(
        signal,
        lambda_revision=0.1,
        lambda_bend=1.0,
    )


def test_registered_dimensions_match_candidate_and_raw() -> None:
    assert runner.PATH_SIGNATURE_WIDTH == 72
    assert runner.PATH_DESCRIPTOR_WIDTH == 104
    assert runner.PATH_INPUT_WIDTH == 121
    assert runner.REGISTERED_RAW_LAGS == 120


def test_level_two_signature_distinguishes_update_order() -> None:
    left = np.zeros((2, runner.PATH_COORDINATES), dtype=np.float64)
    right = np.zeros_like(left)
    left[0, 0], left[1, 1] = 1.0, 1.0
    right[0, 1], right[1, 0] = 1.0, 1.0

    left_signature = runner._path_signature_level_two(left)
    right_signature = runner._path_signature_level_two(right)

    np.testing.assert_array_equal(
        left_signature[: runner.PATH_COORDINATES],
        right_signature[: runner.PATH_COORDINATES],
    )
    assert not np.array_equal(left_signature, right_signature)


def test_design_preserves_geometry_and_adds_complete_path() -> None:
    design = runner.build_revision_path_design(_bundle())

    assert design.inputs["geometry"].shape[1] == runner.GEOMETRY_WIDTH
    assert design.inputs["geometry_last_update"].shape[1] == runner.LAST_UPDATE_WIDTH
    assert design.inputs["geometry_revision_path"].shape[1] == runner.PATH_INPUT_WIDTH
    assert design.inputs["raw_matched"].shape[1] == runner.PATH_INPUT_WIDTH
    np.testing.assert_array_equal(
        design.inputs["geometry_revision_path"][:, : runner.GEOMETRY_WIDTH],
        design.inputs["geometry"],
    )
    assert np.any(design.inputs["geometry_revision_path"][:, runner.GEOMETRY_WIDTH :] != 0.0)
    assert all(not values.flags.writeable for values in design.inputs.values())


def test_path_design_is_prefix_causal() -> None:
    complete = runner.build_revision_path_design(_bundle())
    prefix = runner.build_revision_path_design(_bundle(stop_exclusive=420))
    common_origin = int(prefix.origins[-1])
    complete_row = int(np.flatnonzero(complete.origins == common_origin)[0])
    prefix_row = int(np.flatnonzero(prefix.origins == common_origin)[0])

    for representation in (
        "geometry",
        "geometry_last_update",
        "geometry_revision_path",
        "raw_matched",
    ):
        np.testing.assert_array_equal(
            complete.inputs[representation][complete_row],
            prefix.inputs[representation][prefix_row],
        )


def test_sham_path_preserves_geometry_and_is_deterministic() -> None:
    candidate = runner.build_revision_path_design(_bundle()).inputs["geometry_revision_path"]
    first = runner.sham_path_inputs(candidate, seed=7)
    second = runner.sham_path_inputs(candidate, seed=7)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first[:, : runner.GEOMETRY_WIDTH], candidate[:, : runner.GEOMETRY_WIDTH]
    )
    assert not np.array_equal(
        first[:, runner.GEOMETRY_WIDTH :], candidate[:, runner.GEOMETRY_WIDTH :]
    )


def test_canonical_config_is_loadable_and_uses_new_seeds() -> None:
    config = runner.load_config(
        Path(__file__).parents[1] / "configs" / "forecasting" / "revision_path_kill_test.toml"
    )

    assert config.seeds == (1103, 2207, 3301, 4409, 5501)
    assert config.raw_lags == 120
    assert (config.lambda_revision, config.lambda_bend) == (0.1, 1.0)
