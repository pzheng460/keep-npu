import random

import pytest

from keep_npu.single_npu_controller.random_workload import (
    PROFILE_DEFINITIONS,
    RandomPhaseScheduler,
)


def test_profile_definitions_match_the_public_design():
    assert {
        item.name: (item.cube_range, item.vector_range, item.hbm_range)
        for item in PROFILE_DEFINITIONS
    } == {
        "cube": ((0.85, 1.00), (0.20, 0.45), (0.35, 0.65)),
        "vector": ((0.25, 0.55), (0.80, 1.00), (0.40, 0.70)),
        "hbm": ((0.20, 0.50), (0.35, 0.65), (0.85, 1.00)),
        "balanced": ((0.55, 0.85), (0.55, 0.85), (0.55, 0.85)),
    }


def test_seeded_scheduler_stays_in_bounds_and_never_repeats_profile():
    scheduler = RandomPhaseScheduler(random.Random(460))
    snapshots = []
    previous_profile = None
    for _ in range(1000):
        snapshot = scheduler.advance(1.0)
        if snapshot.profile != previous_profile:
            snapshots.append(snapshot)
            previous_profile = snapshot.profile

    assert len(snapshots) >= 20
    assert all(10.0 <= item.duration <= 60.0 for item in snapshots)
    assert all(
        2.0 <= item.ramp_duration <= min(5.0, item.duration / 4)
        for item in snapshots
    )
    assert all(
        left.profile != right.profile
        for left, right in zip(snapshots, snapshots[1:])
    )
    for item in snapshots:
        profile = next(p for p in PROFILE_DEFINITIONS if p.name == item.profile)
        assert profile.cube_range[0] <= item.target_cube_duty <= profile.cube_range[1]
        assert (
            profile.vector_range[0]
            <= item.target_vector_duty
            <= profile.vector_range[1]
        )
        assert profile.hbm_range[0] <= item.target_hbm_duty <= profile.hbm_range[1]


def test_scheduler_ramps_from_zero_and_freezes_when_advance_is_zero():
    scheduler = RandomPhaseScheduler(random.Random(7))

    initial = scheduler.advance(0.0)
    halfway = scheduler.advance(initial.ramp_duration / 2)
    frozen = scheduler.advance(0.0)

    assert initial.cube_duty == initial.vector_duty == initial.hbm_duty == 0.0
    assert halfway.cube_duty == pytest.approx(halfway.target_cube_duty / 2)
    assert halfway.vector_duty == pytest.approx(halfway.target_vector_duty / 2)
    assert halfway.hbm_duty == pytest.approx(halfway.target_hbm_duty / 2)
    assert frozen == halfway


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_active_elapsed(value):
    scheduler = RandomPhaseScheduler(random.Random(1))

    with pytest.raises(
        ValueError, match="active_seconds must be finite and non-negative"
    ):
        scheduler.advance(value)


def test_large_advance_rolls_over_multiple_phases_without_losing_time():
    scheduler = RandomPhaseScheduler(random.Random(19))

    snapshot = scheduler.advance(180.0)

    assert 0.0 <= snapshot.elapsed < snapshot.duration


def test_profiles_are_selected_from_equal_weight_candidate_sequences():
    class RecordingRandom(random.Random):
        def __init__(self):
            super().__init__(0)
            self.choice_candidates = []

        def choice(self, seq):
            self.choice_candidates.append(tuple(item.name for item in seq))
            return seq[0]

        def uniform(self, low, high):
            return (low + high) / 2

    rng = RecordingRandom()
    scheduler = RandomPhaseScheduler(rng)
    scheduler.advance(36.0)

    assert rng.choice_candidates[0] == ("cube", "vector", "hbm", "balanced")
    assert rng.choice_candidates[1] == ("vector", "hbm", "balanced")
