"""Tests for the bounded causal revisable-tail state machine."""

import numpy as np
import pytest

from vectorchain import (
    RevisableVectorChain,
    VectorChain,
    WorkingTailLimitError,
    WorkingVersion,
)


def test_initial_state_and_configured_limits_are_explicit() -> None:
    chain = RevisableVectorChain()

    assert chain.committed_ == ()
    assert chain.versions_ == ()
    assert chain.events_ == ()
    assert chain.working_version_ is None
    assert chain.working_ == ()
    assert chain.n_samples_ == 0
    assert chain.MAX_WORKING_LINKS == 4
    assert chain.MAX_RAW_INTERVALS == 256


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tolerance": True}, TypeError),
        ({"tolerance": -0.1}, ValueError),
        ({"min_segment_length": 1}, ValueError),
        ({"lambda_revision": True}, TypeError),
        ({"lambda_revision": -0.1}, ValueError),
        ({"lambda_bend": "0.1"}, TypeError),
        ({"lambda_bend": np.inf}, ValueError),
    ],
)
def test_constructor_rejects_invalid_parameters(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        RevisableVectorChain(**kwargs)


def test_each_observation_appends_an_immutable_version_with_stable_ids() -> None:
    chain = RevisableVectorChain(tolerance=0.0)

    versions = chain.fit_transform([0.0, 1.0, 2.0, 2.0])

    assert tuple(version.version for version in versions) == (0, 1, 2, 3)
    assert tuple(version.observed_at for version in versions) == (0, 1, 2, 3)
    assert versions[0].links == ()
    assert versions[2].links[0].link_id == versions[3].links[0].link_id == 0
    assert (versions[3].links[0].start, versions[3].links[0].end) == (0, 2)
    assert (versions[3].links[1].start, versions[3].links[1].end) == (2, 3)
    assert versions[3].links[1].link_id == 1
    assert versions[3].links[1].update_theta == 0.0
    assert versions[3].links[1].update_r == 0.0
    assert versions[2].links[0].end == 2
    assert chain.versions_ == versions


def test_quadratic_revision_preserves_anchors_and_adjusts_internal_joint() -> None:
    chain = RevisableVectorChain(
        tolerance=0.0,
        lambda_revision=0.1,
        lambda_bend=0.1,
    )

    version = chain.fit_transform([0.0, 1.0, 0.0])[-1]

    assert len(version.links) == 2
    assert version.joints[0].value == 0.0
    assert version.joints[-1].value == 0.0
    # For this three-point case the frozen objective has the analytic minimizer 0.52.
    np.testing.assert_allclose(version.joints[1].value, 0.52, rtol=0.0, atol=1e-15)
    assert version.condition_number > 0.0
    assert np.isfinite(version.condition_number)
    assert version.links[0].end_value == version.links[1].start_value


def test_fifth_link_commits_oldest_prior_snapshot_and_keeps_four_working() -> None:
    chain = RevisableVectorChain(tolerance=0.0)
    signal = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    versions = chain.fit_transform(signal)
    committed = chain.committed_

    assert len(committed) == 1
    assert committed[0].link_id == 0
    assert committed[0].committed_at == 5
    assert committed[0].start_value == versions[4].links[0].start_value
    assert committed[0].end_value == versions[4].links[0].end_value
    assert tuple(link.link_id for link in chain.working_) == (1, 2, 3, 4)
    assert len(chain.working_) == chain.MAX_WORKING_LINKS
    assert chain.working_version_ is not None
    assert chain.working_version_.raw_span == 4


def test_batch_wrapper_matches_repeated_online_updates_exactly() -> None:
    signal = np.random.default_rng(11).normal(size=80)
    batch = RevisableVectorChain(tolerance=0.05, lambda_revision=1.0, lambda_bend=0.01)
    online = RevisableVectorChain(tolerance=0.05, lambda_revision=1.0, lambda_bend=0.01)

    batch_versions = batch.fit_transform(signal)
    online_versions = tuple(online.update(float(value)) for value in signal)

    assert online_versions == batch_versions
    assert online.committed_ == batch.committed_
    assert online.events_ == batch.events_


def test_temporal_boundaries_are_exactly_the_causal_vectorchain_proposals() -> None:
    signal = np.random.default_rng(11).normal(size=80)
    proposer = VectorChain(tolerance=0.03, features=("dt", "dy"))
    revisable = RevisableVectorChain(tolerance=0.03)

    for value in signal:
        proposer.update(float(value))
        version = revisable.update(float(value))

        proposed = [(segment.start, segment.end) for segment in proposer.segments_]
        open_boundary = proposer.open_segment_boundary_
        if open_boundary is not None and open_boundary[0] < open_boundary[1]:
            proposed.append(open_boundary)
        represented = [(link.start, link.end) for link in revisable.committed_]
        represented.extend((link.start, link.end) for link in version.links)

        assert represented == proposed


def test_equal_prefix_and_arbitrary_future_preserve_observed_state() -> None:
    prefix = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    left = RevisableVectorChain(tolerance=0.0)
    right = RevisableVectorChain(tolerance=0.0)

    for value in prefix:
        left_version = left.update(value)
        right_version = right.update(value)

    assert left_version == right_version
    committed_at_cut = left.committed_
    assert committed_at_cut == right.committed_
    assert committed_at_cut

    for value in np.random.default_rng(11).normal(size=30):
        left.update(float(value))
    for value in np.random.default_rng(22).normal(loc=100.0, scale=20.0, size=30):
        right.update(float(value))

    count = len(committed_at_cut)
    assert left.committed_[:count] == committed_at_cut
    assert right.committed_[:count] == committed_at_cut


def test_all_versions_obey_geometry_identity_and_bounded_tail_invariants() -> None:
    signal = np.random.default_rng(22).normal(size=120)
    chain = RevisableVectorChain(tolerance=0.03)

    versions = chain.fit_transform(signal)

    for version in versions:
        assert len(version.links) <= chain.MAX_WORKING_LINKS
        assert version.raw_span <= chain.MAX_RAW_INTERVALS
        assert len(version.joints) == len(version.links) + 1
        assert version.joints[-1].sample_index == version.observed_at
        assert version.joints[-1].value == signal[version.observed_at]
        assert np.isfinite(version.condition_number)
        for index, link in enumerate(version.links):
            assert link.dt > 0
            assert link.start_joint_id == version.joints[index].joint_id
            assert link.end_joint_id == version.joints[index + 1].joint_id
            assert link.start_value == version.joints[index].value
            assert link.end_value == version.joints[index + 1].value
            assert np.all(
                np.isfinite(
                    [
                        link.dy,
                        link.theta,
                        link.r,
                        link.update_theta,
                        link.update_r,
                    ]
                )
            )


def test_append_only_history_and_committed_snapshots_never_change() -> None:
    chain = RevisableVectorChain(tolerance=0.0)
    for value in [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]:
        chain.update(value)

    versions_before = chain.versions_
    events_before = chain.events_
    committed_before = chain.committed_

    for value in [100.0, -100.0, 50.0, -50.0]:
        chain.update(value)

    assert chain.versions_[: len(versions_before)] == versions_before
    assert chain.events_[: len(events_before)] == events_before
    assert chain.committed_[: len(committed_before)] == committed_before
    assert {event.kind for event in chain.events_} >= {
        "joint_created",
        "joint_revised",
        "link_created",
        "link_revised",
        "link_committed",
        "version_created",
    }


def test_open_link_limit_fails_explicitly_without_corrupting_stream() -> None:
    chain = RevisableVectorChain(tolerance=1000.0)
    for value in np.arange(257.0):
        chain.update(float(value))

    version_before = chain.working_version_
    events_before = chain.events_

    with pytest.raises(WorkingTailLimitError, match="open link alone"):
        chain.update(257.0)

    assert chain.n_samples_ == 257
    assert chain.working_version_ == version_before
    assert chain.events_ == events_before

    recovered = chain.update(1_000_000.0)
    assert recovered.observed_at == 257
    assert recovered.raw_span == 1
    assert len(chain.committed_) == 1


@pytest.mark.parametrize(
    ("series", "error"),
    [
        ([1.0], ValueError),
        ([[1.0, 2.0]], ValueError),
        (["a", "b"], TypeError),
        ([0.0, np.nan], ValueError),
    ],
)
def test_failed_batch_input_resets_all_state(series: object, error: type[Exception]) -> None:
    chain = RevisableVectorChain()
    chain.fit_transform([0.0, 1.0])

    with pytest.raises(error):
        chain.fit_transform(series)

    assert chain.versions_ == ()
    assert chain.committed_ == ()
    assert chain.events_ == ()
    assert chain.n_samples_ == 0


def test_reset_isolates_the_next_stream() -> None:
    reused = RevisableVectorChain(tolerance=0.0)
    fresh = RevisableVectorChain(tolerance=0.0)
    reused.fit_transform([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    returned = reused.reset()
    signal = [2.0, 3.0, 2.0, 3.0]

    assert returned is reused
    assert reused.fit_transform(signal) == fresh.fit_transform(signal)
    assert reused.committed_ == fresh.committed_
    assert reused.events_ == fresh.events_


def test_update_returns_a_working_version_type() -> None:
    version = RevisableVectorChain().update(1.0)

    assert isinstance(version, WorkingVersion)
