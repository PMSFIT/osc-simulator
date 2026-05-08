"""Integration tests: parse → simulate → verify kinematics."""

from pathlib import Path
from typing import Any

import pytest

from osc_simulator.parser.openscenario import ScenarioParser
from osc_simulator.simulation.engine import SimulationEngine

EXAMPLE = Path(__file__).parent.parent / "examples" / "simple_scenario.xosc"


def _run_all(scenario_path: Path, step_size: float = 0.05) -> list[tuple[float, Any]]:
    scenario = ScenarioParser().parse(scenario_path)
    engine = SimulationEngine(scenario, step_size=step_size)
    return list(engine.run())


def test_frame_count() -> None:
    frames = _run_all(EXAMPLE)
    # 10 s / 0.05 s + 1 final frame = 201
    assert len(frames) == 201


def test_timestamps_monotonic() -> None:
    frames = _run_all(EXAMPLE)
    timestamps = [t for t, _ in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[-1] == pytest.approx(10.0, abs=1e-6)


def test_ego_travels_east() -> None:
    frames = _run_all(EXAMPLE)
    # At t=10 s, Ego should have travelled ~200 m East from origin
    _, gt = frames[-1]
    ego_obj = gt.moving_object[0]
    assert ego_obj.base.position.x == pytest.approx(200.0, rel=0.01)
    assert abs(ego_obj.base.position.y) < 1.0


def test_npc_accelerates_after_3s() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    engine = SimulationEngine(scenario, step_size=0.05)
    frames = list(engine.run())

    # Find NPC object (second moving object)
    t_before, gt_before = next((t, g) for t, g in frames if abs(t - 2.9) < 0.03)
    t_after, gt_after = next((t, g) for t, g in frames if abs(t - 5.5) < 0.03)

    npc_before = next(
        o for o in gt_before.moving_object if o.id.value != gt_before.moving_object[0].id.value
    )
    npc_after = next(
        o for o in gt_after.moving_object if o.id.value != gt_after.moving_object[0].id.value
    )

    speed_before = (npc_before.base.velocity.x**2 + npc_before.base.velocity.y**2) ** 0.5
    speed_after = (npc_after.base.velocity.x**2 + npc_after.base.velocity.y**2) ** 0.5

    assert speed_before == pytest.approx(10.0, abs=0.5)
    assert speed_after > 15.0


def test_ground_truth_has_two_objects() -> None:
    frames = _run_all(EXAMPLE)
    _, gt = frames[0]
    assert len(gt.moving_object) == 2


def test_osi_ground_truth_type() -> None:
    import osi3.osi_groundtruth_pb2 as osi_gt

    frames = _run_all(EXAMPLE)
    _, gt = frames[0]
    assert isinstance(gt, osi_gt.GroundTruth)
