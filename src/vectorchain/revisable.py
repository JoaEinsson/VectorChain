"""Bounded causal revision of a provisional vector-chain tail."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from numbers import Real
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorchain.core import VectorChain

TailEventKind = Literal[
    "joint_created",
    "joint_revised",
    "link_created",
    "link_revised",
    "link_committed",
    "version_created",
]


class WorkingTailLimitError(RuntimeError):
    """Raised when the open link alone exceeds the bounded raw lag."""


class RevisionSolverError(RuntimeError):
    """Raised when the deterministic quadratic revision cannot be solved."""


@dataclass(frozen=True, slots=True)
class WorkingJoint:
    """One joint in an immutable version of the provisional tail."""

    joint_id: int
    sample_index: int
    value: float
    created_at: int


@dataclass(frozen=True, slots=True)
class WorkingLink:
    """One link in an immutable version of the provisional tail."""

    link_id: int
    start_joint_id: int
    end_joint_id: int
    start: int
    end: int
    start_value: float
    end_value: float
    created_at: int
    update_theta: float
    update_r: float

    @property
    def dt(self) -> int:
        """Return the link duration in raw sample intervals."""

        return self.end - self.start

    @property
    def dy(self) -> float:
        """Return the link vertical displacement."""

        return self.end_value - self.start_value

    @property
    def theta(self) -> float:
        """Return the link angle in radians."""

        return atan2(self.dy, self.dt)

    @property
    def r(self) -> float:
        """Return the link length in the mixed time-amplitude coordinates."""

        return hypot(self.dt, self.dy)


@dataclass(frozen=True, slots=True)
class CommittedLink:
    """One immutable link removed from the bounded working tail."""

    link_id: int
    start_joint_id: int
    end_joint_id: int
    start: int
    end: int
    start_value: float
    end_value: float
    created_at: int
    committed_at: int

    @property
    def dt(self) -> int:
        """Return the link duration in raw sample intervals."""

        return self.end - self.start

    @property
    def dy(self) -> float:
        """Return the link vertical displacement at commitment."""

        return self.end_value - self.start_value

    @property
    def theta(self) -> float:
        """Return the link angle at commitment."""

        return atan2(self.dy, self.dt)

    @property
    def r(self) -> float:
        """Return the link length at commitment."""

        return hypot(self.dt, self.dy)


@dataclass(frozen=True, slots=True)
class WorkingVersion:
    """Immutable, causal snapshot of the complete provisional tail."""

    version: int
    observed_at: int
    joints: tuple[WorkingJoint, ...]
    links: tuple[WorkingLink, ...]
    condition_number: float

    @property
    def raw_span(self) -> int:
        """Return the number of raw intervals covered by this tail."""

        if len(self.joints) < 2:
            return 0
        return self.joints[-1].sample_index - self.joints[0].sample_index


@dataclass(frozen=True, slots=True)
class TailEvent:
    """Append-only audit event produced by a tail transition."""

    kind: TailEventKind
    observed_at: int
    version: int
    target_id: int | None


class RevisableVectorChain:
    """Maintain a causal, versioned and bounded revisable vector tail.

    Temporal boundaries come from the existing causal :class:`VectorChain` state
    machine. Only provisional joint ordinates are revised. The oldest complete
    link is committed before a transition would expose more than four working
    links or 256 raw intervals; committed links are immutable snapshots.

    Parameters
    ----------
    tolerance:
        Absolute chord-error threshold used only to propose temporal boundaries.
    min_segment_length:
        Minimum point count used by the causal boundary proposer.
    lambda_revision:
        Non-negative penalty on changes to existing internal joint ordinates.
    lambda_bend:
        Non-negative penalty on differences between adjacent link slopes.

    Notes
    -----
    ``update`` records one immutable :class:`WorkingVersion` per accepted
    observation. A new link has zero ``update_theta`` and ``update_r``; later
    versions compare geometry only against the same stable link identity.
    """

    MAX_WORKING_LINKS: ClassVar[int] = 4
    MAX_RAW_INTERVALS: ClassVar[int] = 256

    def __init__(
        self,
        tolerance: float = 0.03,
        min_segment_length: int = 2,
        lambda_revision: float = 0.1,
        lambda_bend: float = 0.1,
    ) -> None:
        self.tolerance = VectorChain._validate_tolerance(tolerance)
        self.min_segment_length = VectorChain._validate_min_segment_length(min_segment_length)
        self.lambda_revision = self._validate_penalty(lambda_revision, "lambda_revision")
        self.lambda_bend = self._validate_penalty(lambda_bend, "lambda_bend")
        self.reset()

    @property
    def committed_(self) -> tuple[CommittedLink, ...]:
        """Return an immutable snapshot of every committed link."""

        return tuple(self._committed)

    @property
    def versions_(self) -> tuple[WorkingVersion, ...]:
        """Return the append-only history of provisional tail versions."""

        return tuple(self._versions)

    @property
    def events_(self) -> tuple[TailEvent, ...]:
        """Return the append-only transition audit log."""

        return tuple(self._events)

    @property
    def working_version_(self) -> WorkingVersion | None:
        """Return the latest provisional tail version, if any."""

        if not self._versions:
            return None
        return self._versions[-1]

    @property
    def working_(self) -> tuple[WorkingLink, ...]:
        """Return the links in the latest provisional tail version."""

        latest = self.working_version_
        return () if latest is None else latest.links

    def reset(self) -> RevisableVectorChain:
        """Clear committed, provisional and audit state."""

        self._segmenter = self._new_segmenter()
        self._values: list[float] = []
        self._times: list[int] = []
        self._joint_ids: list[int] = []
        self._joint_created_at: list[int] = []
        self._link_ids: list[int] = []
        self._link_created_at: list[int] = []
        self._committed: list[CommittedLink] = []
        self._versions: list[WorkingVersion] = []
        self._events: list[TailEvent] = []
        self._next_joint_id = 0
        self._next_link_id = 0
        self.n_samples_ = 0
        return self

    def update(self, value: float) -> WorkingVersion:
        """Consume one observation and append its causal working-tail version.

        If an unbroken open link would exceed 256 raw intervals, the observation
        is rejected with :class:`WorkingTailLimitError` and the prior state is
        preserved. This explicit failure avoids inventing an unapproved boundary.
        """

        observed_at = self.n_samples_
        emitted = self._segmenter.update(value)
        sample = float(value)

        open_boundary = self._segmenter.open_segment_boundary_
        if (
            open_boundary is not None
            and open_boundary[1] - open_boundary[0] > self.MAX_RAW_INTERVALS
        ):
            self._restore_segmenter()
            msg = "the open link alone exceeds the 256-interval working-tail limit"
            raise WorkingTailLimitError(msg)

        self._values.append(sample)
        self.n_samples_ += 1
        version = len(self._versions)

        if observed_at == 0:
            self._append_initial_joint(observed_at, version)
            return self._record_version(observed_at, version)

        if observed_at == 1:
            self._append_first_link(observed_at, version)
        elif emitted:
            self._append_new_open_link(observed_at, version)
        else:
            self._times[-1] = observed_at

        while len(self._link_ids) > self.MAX_WORKING_LINKS:
            self._commit_oldest(observed_at, version)
        while self._times[-1] - self._times[0] > self.MAX_RAW_INTERVALS:
            self._commit_oldest(observed_at, version)

        return self._record_version(observed_at, version)

    def fit_transform(self, x: ArrayLike) -> tuple[WorkingVersion, ...]:
        """Reset and consume a finite one-dimensional series through ``update``.

        The returned tuple contains exactly the same immutable versions produced
        by repeated online calls. The final provisional tail is intentionally not
        committed merely because the finite input ended.
        """

        self.reset()
        try:
            values = self._validate_series(x)
            for value in values:
                self.update(float(value))
        except Exception:
            self.reset()
            raise
        return self.versions_

    @staticmethod
    def _validate_penalty(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            msg = f"{name} must be a finite non-negative real number"
            raise TypeError(msg)
        validated = float(value)
        if not np.isfinite(validated) or validated < 0.0:
            msg = f"{name} must be finite and non-negative"
            raise ValueError(msg)
        return validated

    @staticmethod
    def _validate_series(x: ArrayLike) -> NDArray[np.float64]:
        try:
            values = np.asarray(x)
        except (TypeError, ValueError) as error:
            msg = "x must be a one-dimensional numeric series"
            raise ValueError(msg) from error
        if values.ndim != 1 or values.size < 2:
            msg = "x must be a one-dimensional series with at least two observations"
            raise ValueError(msg)
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            msg = "x must contain real numeric observations"
            raise TypeError(msg)
        converted = values.astype(np.float64, copy=False)
        if not np.all(np.isfinite(converted)):
            msg = "x must contain finite real observations"
            raise ValueError(msg)
        return converted

    def _new_segmenter(self) -> VectorChain:
        return VectorChain(
            tolerance=self.tolerance,
            min_segment_length=self.min_segment_length,
            features=("dt", "dy"),
        )

    def _restore_segmenter(self) -> None:
        self._segmenter = self._new_segmenter()
        for previous in self._values:
            self._segmenter.update(previous)

    def _append_initial_joint(self, observed_at: int, version: int) -> None:
        joint_id = self._allocate_joint_id()
        self._times.append(observed_at)
        self._joint_ids.append(joint_id)
        self._joint_created_at.append(observed_at)
        self._events.append(TailEvent("joint_created", observed_at, version, joint_id))

    def _append_first_link(self, observed_at: int, version: int) -> None:
        joint_id = self._allocate_joint_id()
        link_id = self._allocate_link_id()
        self._times.append(observed_at)
        self._joint_ids.append(joint_id)
        self._joint_created_at.append(observed_at)
        self._link_ids.append(link_id)
        self._link_created_at.append(observed_at)
        self._events.extend(
            (
                TailEvent("joint_created", observed_at, version, joint_id),
                TailEvent("link_created", observed_at, version, link_id),
            )
        )

    def _append_new_open_link(self, observed_at: int, version: int) -> None:
        joint_id = self._allocate_joint_id()
        link_id = self._allocate_link_id()
        self._times.append(observed_at)
        self._joint_ids.append(joint_id)
        self._joint_created_at.append(observed_at)
        self._link_ids.append(link_id)
        self._link_created_at.append(observed_at)
        self._events.extend(
            (
                TailEvent("joint_created", observed_at, version, joint_id),
                TailEvent("link_created", observed_at, version, link_id),
            )
        )

    def _allocate_joint_id(self) -> int:
        joint_id = self._next_joint_id
        self._next_joint_id += 1
        return joint_id

    def _allocate_link_id(self) -> int:
        link_id = self._next_link_id
        self._next_link_id += 1
        return link_id

    def _commit_oldest(self, observed_at: int, version: int) -> None:
        if len(self._link_ids) < 2:
            msg = "internal error: cannot commit the only open working link"
            raise WorkingTailLimitError(msg)
        previous = self.working_version_
        if previous is None:
            msg = "internal error: no prior version is available for commitment"
            raise RuntimeError(msg)
        link_id = self._link_ids[0]
        prior_link = next(link for link in previous.links if link.link_id == link_id)
        committed = CommittedLink(
            link_id=prior_link.link_id,
            start_joint_id=prior_link.start_joint_id,
            end_joint_id=prior_link.end_joint_id,
            start=prior_link.start,
            end=prior_link.end,
            start_value=prior_link.start_value,
            end_value=prior_link.end_value,
            created_at=prior_link.created_at,
            committed_at=observed_at,
        )
        self._committed.append(committed)
        self._events.append(TailEvent("link_committed", observed_at, version, link_id))
        del self._times[0]
        del self._joint_ids[0]
        del self._joint_created_at[0]
        del self._link_ids[0]
        del self._link_created_at[0]

    def _record_version(self, observed_at: int, version: int) -> WorkingVersion:
        ordinates, condition_number = self._solve_ordinates()
        previous = self.working_version_
        old_joints = (
            {} if previous is None else {joint.joint_id: joint for joint in previous.joints}
        )
        old_links = {} if previous is None else {link.link_id: link for link in previous.links}

        joints = tuple(
            WorkingJoint(joint_id, time, float(value), created_at)
            for joint_id, time, value, created_at in zip(
                self._joint_ids, self._times, ordinates, self._joint_created_at, strict=True
            )
        )
        links: list[WorkingLink] = []
        for index, (link_id, created_at) in enumerate(
            zip(self._link_ids, self._link_created_at, strict=True)
        ):
            provisional = WorkingLink(
                link_id=link_id,
                start_joint_id=self._joint_ids[index],
                end_joint_id=self._joint_ids[index + 1],
                start=self._times[index],
                end=self._times[index + 1],
                start_value=float(ordinates[index]),
                end_value=float(ordinates[index + 1]),
                created_at=created_at,
                update_theta=0.0,
                update_r=0.0,
            )
            prior = old_links.get(link_id)
            if prior is not None:
                provisional = WorkingLink(
                    link_id=provisional.link_id,
                    start_joint_id=provisional.start_joint_id,
                    end_joint_id=provisional.end_joint_id,
                    start=provisional.start,
                    end=provisional.end,
                    start_value=provisional.start_value,
                    end_value=provisional.end_value,
                    created_at=provisional.created_at,
                    update_theta=provisional.theta - prior.theta,
                    update_r=provisional.r - prior.r,
                )
                if provisional.update_theta != 0.0 or provisional.update_r != 0.0:
                    self._events.append(TailEvent("link_revised", observed_at, version, link_id))
            links.append(provisional)

        for joint in joints:
            prior_joint = old_joints.get(joint.joint_id)
            if prior_joint is not None and (
                joint.sample_index != prior_joint.sample_index or joint.value != prior_joint.value
            ):
                self._events.append(
                    TailEvent("joint_revised", observed_at, version, joint.joint_id)
                )

        snapshot = WorkingVersion(
            version=version,
            observed_at=observed_at,
            joints=joints,
            links=tuple(links),
            condition_number=condition_number,
        )
        self._versions.append(snapshot)
        self._events.append(TailEvent("version_created", observed_at, version, None))
        return snapshot

    def _solve_ordinates(self) -> tuple[NDArray[np.float64], float]:
        if len(self._times) == 1:
            return np.asarray([self._values[0]], dtype=np.float64), 1.0

        times = np.asarray(self._times, dtype=np.int64)
        n_links = len(times) - 1
        ordinates = np.empty(times.size, dtype=np.float64)
        ordinates[0] = self._committed[-1].end_value if self._committed else self._values[times[0]]
        ordinates[-1] = self._values[times[-1]]
        if n_links == 1:
            return ordinates, 1.0

        sample_times = np.arange(times[0], times[-1] + 1, dtype=np.int64)
        raw = np.asarray(self._values[times[0] : times[-1] + 1], dtype=np.float64)
        basis = np.zeros((sample_times.size, times.size), dtype=np.float64)
        link_indices = np.searchsorted(times[1:], sample_times, side="left")
        left_times = times[link_indices]
        right_times = times[link_indices + 1]
        fractions = (sample_times - left_times) / (right_times - left_times)
        rows = np.arange(sample_times.size)
        basis[rows, link_indices] = 1.0 - fractions
        basis[rows, link_indices + 1] = fractions

        hessian = basis.T @ basis / sample_times.size
        linear = basis.T @ raw / sample_times.size

        bend = np.zeros((n_links - 1, times.size), dtype=np.float64)
        for row, joint_index in enumerate(range(1, n_links)):
            left_dt = times[joint_index] - times[joint_index - 1]
            right_dt = times[joint_index + 1] - times[joint_index]
            bend[row, joint_index - 1] = 1.0 / left_dt
            bend[row, joint_index] = -(1.0 / left_dt + 1.0 / right_dt)
            bend[row, joint_index + 1] = 1.0 / right_dt
        hessian += self.lambda_bend * (bend.T @ bend) / bend.shape[0]

        internal = np.arange(1, n_links)
        previous = self.working_version_
        previous_values = (
            {} if previous is None else {joint.joint_id: joint.value for joint in previous.joints}
        )
        revision_targets = [
            (position, previous_values[joint_id])
            for position, joint_id in enumerate(self._joint_ids[1:-1], start=1)
            if joint_id in previous_values
        ]
        if revision_targets:
            weight = self.lambda_revision / len(revision_targets)
            for position, target in revision_targets:
                hessian[position, position] += weight
                linear[position] += weight * target

        anchors = np.asarray([0, n_links])
        anchor_values = ordinates[anchors]
        system = hessian[np.ix_(internal, internal)]
        rhs = linear[internal] - hessian[np.ix_(internal, anchors)] @ anchor_values
        condition_number = float(np.linalg.cond(system))
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as error:
            msg = "the quadratic working-tail revision is singular"
            raise RevisionSolverError(msg) from error
        if not np.isfinite(condition_number) or not np.all(np.isfinite(solution)):
            msg = "the quadratic working-tail revision produced a non-finite solution"
            raise RevisionSolverError(msg)
        ordinates[internal] = solution
        return ordinates, condition_number
