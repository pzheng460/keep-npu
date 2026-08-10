"""Pure phase scheduling for the random Ascend workload."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

DutyRange = Tuple[float, float]
DutyTargets = Tuple[float, float, float]


@dataclass(frozen=True)
class ProfileDefinition:
    """Duty-cycle ranges for one recognizable workload phase."""

    name: str
    cube_range: DutyRange
    vector_range: DutyRange
    hbm_range: DutyRange


@dataclass(frozen=True)
class PhaseSnapshot:
    """One immutable view of the current interpolated phase."""

    profile: str
    cube_duty: float
    vector_duty: float
    hbm_duty: float
    target_cube_duty: float
    target_vector_duty: float
    target_hbm_duty: float
    duration: float
    ramp_duration: float
    elapsed: float


@dataclass(frozen=True)
class _Phase:
    profile: ProfileDefinition
    targets: DutyTargets
    duration: float
    ramp_duration: float


PROFILE_DEFINITIONS = (
    ProfileDefinition("cube", (0.85, 1.00), (0.20, 0.45), (0.35, 0.65)),
    ProfileDefinition("vector", (0.25, 0.55), (0.80, 1.00), (0.40, 0.70)),
    ProfileDefinition("hbm", (0.20, 0.50), (0.35, 0.65), (0.85, 1.00)),
    ProfileDefinition("balanced", (0.55, 0.85), (0.55, 0.85), (0.55, 0.85)),
)


class RandomPhaseScheduler:
    """Advance a reproducible virtual clock through random workload phases."""

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng if rng is not None else random.Random()
        self._previous_profile: Optional[str] = None
        self._previous_targets: DutyTargets = (0.0, 0.0, 0.0)
        self._elapsed = 0.0
        self._phase = self._select_phase()

    def _select_phase(self) -> _Phase:
        candidates = tuple(
            profile
            for profile in PROFILE_DEFINITIONS
            if profile.name != self._previous_profile
        )
        profile = self._rng.choice(candidates)
        duration = self._rng.uniform(10.0, 60.0)
        ramp_duration = min(self._rng.uniform(2.0, 5.0), duration / 4)
        targets = (
            self._rng.uniform(*profile.cube_range),
            self._rng.uniform(*profile.vector_range),
            self._rng.uniform(*profile.hbm_range),
        )
        return _Phase(
            profile=profile,
            targets=targets,
            duration=duration,
            ramp_duration=ramp_duration,
        )

    def advance(self, active_seconds: float) -> PhaseSnapshot:
        """Advance only active time; zero freezes the phase and ramp clocks."""
        if (
            isinstance(active_seconds, bool)
            or not isinstance(active_seconds, (int, float))
            or not math.isfinite(active_seconds)
            or active_seconds < 0
        ):
            raise ValueError("active_seconds must be finite and non-negative")
        self._elapsed += float(active_seconds)
        while self._elapsed >= self._phase.duration:
            self._elapsed -= self._phase.duration
            self._previous_profile = self._phase.profile.name
            self._previous_targets = self._phase.targets
            self._phase = self._select_phase()

        ramp_fraction = min(1.0, self._elapsed / self._phase.ramp_duration)
        duties = tuple(
            previous + (target - previous) * ramp_fraction
            for previous, target in zip(self._previous_targets, self._phase.targets)
        )
        return PhaseSnapshot(
            profile=self._phase.profile.name,
            cube_duty=duties[0],
            vector_duty=duties[1],
            hbm_duty=duties[2],
            target_cube_duty=self._phase.targets[0],
            target_vector_duty=self._phase.targets[1],
            target_hbm_duty=self._phase.targets[2],
            duration=self._phase.duration,
            ramp_duration=self._phase.ramp_duration,
            elapsed=self._elapsed,
        )
