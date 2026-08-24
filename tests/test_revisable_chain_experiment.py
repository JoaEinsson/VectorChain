"""Tests for the pre-registered K7 signals and matched input designs."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "revisable_chain.py"
MODULE_SPEC = importlib.util.spec_from_file_location("vectorchain_revisable_chain", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("could not load revisable-chain experiment module")
revisable_chain = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = revisable_chain
MODULE_SPEC.loader.exec_module(revisable_chain)


@pytest.fixture(scope="module")
def frequency_bundle() -> revisable_chain.K7DesignBundle:
    signal = revisable_chain.generate_k7_signal(
        "frequency_modulation",
        seed=11,
        n_points=256,
    )
    return revisable_chain.build_k7_designs(
        signal,
        lambda_revision=0.1,
        lambda_bend=0.1,
    )


def test_registered_development_scope_is_exact() -> None:
    assert revisable_chain.DEVELOPMENT_SEEDS == (11, 22)
    assert revisable_chain.MECHANISM_NAMES == (
        "frequency_modulation",
        "baseline_modulation",
        "crest_asymmetry_modulation",
    )
    assert revisable_chain.REPRESENTATION_NAMES == (
        "immutable_absolute",
        "revisable_absolute",
        "revisable_spatial",
        "revisable_temporal",
        "raw_matched",
        "persistence",
    )
    assert revisable_chain.HORIZONS == (1, 8, 32)
    assert revisable_chain.N_POINTS == 4096
    assert revisable_chain.NOISE_STD == 0.02
    assert revisable_chain.N_PREDICTIVE_PARAMETERS == 54


@pytest.mark.parametrize(
    ("mechanism", "latent_name"),
    [
        ("frequency_modulation", "f"),
        ("baseline_modulation", "mu"),
        ("crest_asymmetry_modulation", "kappa"),
    ],
)
@pytest.mark.parametrize("seed", revisable_chain.DEVELOPMENT_SEEDS)
def test_k7_signals_are_reproducible_finite_and_read_only(
    mechanism: str, latent_name: str, seed: int
) -> None:
    first = revisable_chain.generate_k7_signal(mechanism, seed=seed, n_points=128)
    second = revisable_chain.generate_k7_signal(mechanism, seed=seed, n_points=128)

    assert first.mechanism == mechanism
    assert first.seed == seed
    assert first.latent_name == latent_name
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.latent_coordinate, second.latent_coordinate)
    np.testing.assert_array_equal(first.latent_derivative, second.latent_derivative)
    for values in (first.values, first.latent_coordinate, first.latent_derivative):
        assert values.shape == (128,)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        assert not values.flags.writeable


def test_frequency_modulation_matches_right_endpoint_phase_recurrence() -> None:
    signal = revisable_chain.generate_k7_signal(
        "frequency_modulation",
        seed=11,
        n_points=13,
        noise_std=0.0,
    )
    normalized_time = np.linspace(0.0, 1.0, 13)
    modulation_phase = 2.0 * np.pi * 3.0 * normalized_time
    frequency = 20.0 - 12.0 * np.cos(modulation_phase)
    phase = np.zeros(13)
    phase[1:] = np.cumsum(2.0 * np.pi * frequency[1:] / 12.0)

    np.testing.assert_allclose(signal.values, np.sin(phase), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(signal.latent_coordinate, frequency, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        signal.latent_derivative,
        72.0 * np.pi * np.sin(modulation_phase),
        rtol=0.0,
        atol=1e-14,
    )
    assert signal.latent_coordinate[0] == 8.0
    assert signal.latent_coordinate[2] == 32.0


def test_baseline_and_crest_conditions_isolate_only_the_registered_coordinate() -> None:
    baseline = revisable_chain.generate_k7_signal(
        "baseline_modulation", seed=11, n_points=13, noise_std=0.0
    )
    crest = revisable_chain.generate_k7_signal(
        "crest_asymmetry_modulation", seed=11, n_points=13, noise_std=0.0
    )
    normalized_time = np.linspace(0.0, 1.0, 13)
    modulation_phase = 2.0 * np.pi * 3.0 * normalized_time
    carrier_phase = 2.0 * np.pi * 16.0 * normalized_time
    baseline_coordinate = 1.0 - np.cos(modulation_phase)
    kappa = 0.225 * baseline_coordinate
    expected_crest = (np.sin(carrier_phase) + kappa * np.sin(2.0 * carrier_phase + np.pi / 4.0)) / (
        1.0 + np.abs(kappa)
    )

    np.testing.assert_allclose(
        baseline.values,
        baseline_coordinate + np.sin(carrier_phase),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(crest.values, expected_crest, rtol=0.0, atol=1e-14)
    assert np.min(baseline.latent_coordinate) == pytest.approx(0.0, abs=1e-15)
    assert np.max(baseline.latent_coordinate) == pytest.approx(2.0)
    assert np.min(crest.latent_coordinate) == pytest.approx(0.0, abs=1e-15)
    assert np.max(crest.latent_coordinate) == pytest.approx(0.45)


def test_signal_noise_uses_only_the_explicit_generator_seed() -> None:
    previous_state = np.random.get_state()
    try:
        np.random.seed(1234)
        expected = np.random.random()
        np.random.seed(1234)

        left = revisable_chain.generate_k7_signal("baseline_modulation", seed=11, n_points=64)
        right = revisable_chain.generate_k7_signal("baseline_modulation", seed=22, n_points=64)

        assert not np.array_equal(left.values, right.values)
        assert np.random.random() == expected
    finally:
        np.random.set_state(previous_state)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"mechanism": "combined", "seed": 11}, ValueError),
        ({"mechanism": "frequency_modulation", "seed": True}, TypeError),
        ({"mechanism": "frequency_modulation", "seed": -1}, ValueError),
        ({"mechanism": "frequency_modulation", "seed": 11, "n_points": True}, TypeError),
        ({"mechanism": "frequency_modulation", "seed": 11, "n_points": 1}, ValueError),
        ({"mechanism": "frequency_modulation", "seed": 11, "noise_std": True}, TypeError),
        ({"mechanism": "frequency_modulation", "seed": 11, "noise_std": -0.1}, ValueError),
        ({"mechanism": "frequency_modulation", "seed": 11, "noise_std": np.inf}, ValueError),
    ],
)
def test_k7_signal_rejects_unregistered_or_invalid_inputs(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        revisable_chain.generate_k7_signal(**kwargs)


def test_all_representations_share_origins_targets_and_registered_payload(
    frequency_bundle: revisable_chain.K7DesignBundle,
) -> None:
    bundle = frequency_bundle
    n_examples = bundle.origins.size

    assert n_examples > 0
    assert tuple(design.name for design in bundle.representations) == (
        revisable_chain.REPRESENTATION_NAMES
    )
    assert bundle.target_indices.shape == (n_examples, 3)
    assert bundle.targets.shape == (n_examples, 3)
    assert bundle.link_ids.shape == (n_examples, 4)
    assert bundle.link_created_at.shape == (n_examples, 4)
    assert bundle.boundaries.shape == (n_examples, 4, 2)
    assert bundle.condition_numbers.shape == (n_examples,)
    assert np.all(bundle.origins % revisable_chain.ORIGIN_STRIDE == 0)
    assert np.all(bundle.origins >= revisable_chain.RAW_MATCHED_STEPS)
    np.testing.assert_array_equal(
        bundle.target_indices,
        bundle.origins[:, np.newaxis] + np.asarray(revisable_chain.HORIZONS),
    )
    np.testing.assert_array_equal(
        bundle.targets,
        bundle.signal.values[bundle.target_indices]
        - bundle.signal.values[bundle.origins, np.newaxis],
    )

    for design in bundle.representations[:-1]:
        assert design.inputs.shape == (n_examples, 17)
        assert len(design.feature_names) == 17
        assert design.scalar_elements == 17
        assert design.predictive_parameters == 54
        assert design.input_steps == (16 if design.name == "raw_matched" else 4)
        assert np.all(np.isfinite(design.inputs))
        assert not design.inputs.flags.writeable
    persistence = bundle.representation("persistence")
    assert persistence.inputs.shape == (n_examples, 0)
    assert persistence.feature_names == ()
    assert persistence.input_steps == 0
    assert persistence.scalar_elements == 0
    assert persistence.predictive_parameters == 0

    for array in (
        bundle.origins,
        bundle.target_indices,
        bundle.targets,
        bundle.link_ids,
        bundle.link_created_at,
        bundle.boundaries,
        bundle.condition_numbers,
    ):
        assert not array.flags.writeable


def test_registered_feature_values_match_the_causal_working_version(
    frequency_bundle: revisable_chain.K7DesignBundle,
) -> None:
    bundle = frequency_bundle
    row = bundle.origins.size // 2
    origin = int(bundle.origins[row])
    version = bundle.versions[origin]
    assert version.observed_at == origin
    assert len(version.links) == 4

    immutable_expected = []
    absolute_expected = []
    spatial_expected = []
    temporal_expected = []
    previous_theta: float | None = None
    previous_r: float | None = None
    for link in version.links:
        raw_dy = bundle.signal.values[link.end] - bundle.signal.values[link.start]
        immutable_expected.extend(
            (link.dt, raw_dy, np.arctan2(raw_dy, link.dt), np.hypot(link.dt, raw_dy))
        )
        absolute_expected.extend((link.dt, link.dy, link.theta, link.r))
        spatial_expected.extend(
            (
                link.dt,
                link.dy,
                0.0 if previous_theta is None else link.theta - previous_theta,
                0.0 if previous_r is None else link.r - previous_r,
            )
        )
        temporal_expected.extend((link.dt, link.dy, link.update_theta, link.update_r))
        previous_theta = link.theta
        previous_r = link.r
    anchor = bundle.signal.values[origin]
    immutable_expected.append(anchor)
    absolute_expected.append(anchor)
    spatial_expected.append(anchor)
    temporal_expected.append(anchor)

    np.testing.assert_allclose(
        bundle.representation("immutable_absolute").inputs[row], immutable_expected
    )
    np.testing.assert_allclose(
        bundle.representation("revisable_absolute").inputs[row], absolute_expected
    )
    np.testing.assert_allclose(
        bundle.representation("revisable_spatial").inputs[row], spatial_expected
    )
    np.testing.assert_allclose(
        bundle.representation("revisable_temporal").inputs[row], temporal_expected
    )
    np.testing.assert_array_equal(
        bundle.representation("raw_matched").inputs[row, :-1],
        np.diff(bundle.signal.values[origin - 16 : origin + 1]),
    )
    assert bundle.representation("raw_matched").inputs[row, -1] == anchor
    assert bundle.representation("revisable_spatial").inputs[row, 2] == 0.0
    assert bundle.representation("revisable_spatial").inputs[row, 3] == 0.0


def test_regularizers_change_only_revised_designs_not_shared_structure() -> None:
    signal = revisable_chain.generate_k7_signal("crest_asymmetry_modulation", seed=11, n_points=192)
    weak = revisable_chain.build_k7_designs(signal, lambda_revision=0.01, lambda_bend=0.01)
    strong = revisable_chain.build_k7_designs(signal, lambda_revision=1.0, lambda_bend=1.0)

    np.testing.assert_array_equal(weak.origins, strong.origins)
    np.testing.assert_array_equal(weak.target_indices, strong.target_indices)
    np.testing.assert_array_equal(weak.targets, strong.targets)
    np.testing.assert_array_equal(weak.link_ids, strong.link_ids)
    np.testing.assert_array_equal(weak.boundaries, strong.boundaries)
    np.testing.assert_array_equal(
        weak.representation("immutable_absolute").inputs,
        strong.representation("immutable_absolute").inputs,
    )
    np.testing.assert_array_equal(
        weak.representation("raw_matched").inputs,
        strong.representation("raw_matched").inputs,
    )
    assert not np.array_equal(
        weak.representation("revisable_absolute").inputs,
        strong.representation("revisable_absolute").inputs,
    )


def test_arbitrary_future_change_does_not_change_prior_design_rows() -> None:
    prefix_end = 160
    base = revisable_chain.generate_k7_signal("baseline_modulation", seed=22, n_points=256)
    modified_values = base.values.copy()
    modified_values[prefix_end + 1 :] = np.random.default_rng(22).normal(
        loc=100.0,
        scale=20.0,
        size=modified_values.size - prefix_end - 1,
    )
    modified = replace(base, values=modified_values)
    left = revisable_chain.build_k7_designs(base, lambda_revision=0.1, lambda_bend=0.1)
    right = revisable_chain.build_k7_designs(modified, lambda_revision=0.1, lambda_bend=0.1)

    left_rows = {
        int(origin): row for row, origin in enumerate(left.origins) if origin <= prefix_end
    }
    right_rows = {
        int(origin): row for row, origin in enumerate(right.origins) if origin <= prefix_end
    }
    assert left_rows.keys() == right_rows.keys()
    for origin in left_rows:
        left_row = left_rows[origin]
        right_row = right_rows[origin]
        np.testing.assert_array_equal(left.link_ids[left_row], right.link_ids[right_row])
        np.testing.assert_array_equal(left.boundaries[left_row], right.boundaries[right_row])
        for name in revisable_chain.TRAINED_REPRESENTATION_NAMES:
            np.testing.assert_array_equal(
                left.representation(name).inputs[left_row],
                right.representation(name).inputs[right_row],
            )
        if origin + max(revisable_chain.HORIZONS) <= prefix_end:
            np.testing.assert_array_equal(left.targets[left_row], right.targets[right_row])


def test_design_builder_rejects_invalid_or_ineligible_signal() -> None:
    with pytest.raises(TypeError, match="K7Signal"):
        revisable_chain.build_k7_designs(  # type: ignore[arg-type]
            np.arange(64.0), lambda_revision=0.1, lambda_bend=0.1
        )

    valid = revisable_chain.generate_k7_signal(
        "baseline_modulation", seed=11, n_points=64, noise_std=0.0
    )
    malformed = replace(valid, latent_coordinate=np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="latent arrays"):
        revisable_chain.build_k7_designs(malformed, lambda_revision=0.1, lambda_bend=0.1)

    linear = replace(valid, values=np.zeros(64, dtype=np.float64))
    with pytest.raises(ValueError, match="no origins eligible"):
        revisable_chain.build_k7_designs(linear, lambda_revision=0.1, lambda_bend=0.1)


def test_unknown_representation_lookup_fails_explicitly(
    frequency_bundle: revisable_chain.K7DesignBundle,
) -> None:
    with pytest.raises(KeyError, match="combined"):
        frequency_bundle.representation("combined")
