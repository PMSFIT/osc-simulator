"""Tests for FollowTrajectoryAction parsing and simulation."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

from osc_simulator.parser.openscenario import (
    FollowTrajectoryAction,
    ScenarioParser,
    TrajectoryVertex,
)
from osc_simulator.simulation.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xosc(vertices: list[tuple[float, float, float, float]]) -> str:
    """Return a minimal .xosc string with one vehicle following a polyline.

    *vertices* is a list of (time, x, y, z) tuples.
    The entity is also teleported to the first vertex in Init (matching the
    pattern used by osc-validation's osi2osc converter).
    The stop trigger fires just after the last vertex time.
    """
    end_time = vertices[-1][0] + 0.1
    init_x, init_y, init_z = vertices[0][1], vertices[0][2], vertices[0][3]

    vertex_xml = "\n".join(
        f"""            <Vertex time="{t}">
              <Position>
                <WorldPosition x="{x}" y="{y}" z="{z}" h="0.0" p="0.0" r="0.0"/>
              </Position>
            </Vertex>"""
        for t, x, y, z in vertices
    )

    return textwrap.dedent(f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2024-01-01T00:00:00"
              description="Trajectory test" author="test"/>
  <ParameterDeclarations/>
  <CatalogLocations/>
  <RoadNetwork>
    <LogicFile filepath=""/>
    <SceneGraphFile filepath=""/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoVehicle" vehicleCategory="car">
        <BoundingBox>
          <Center x="1.4" y="0.0" z="0.75"/>
          <Dimensions width="2.0" length="4.8" height="1.5"/>
        </BoundingBox>
        <Performance maxSpeed="70" maxAcceleration="10" maxDeceleration="10"/>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8"
                     positionX="2.8" positionZ="0.3"/>
          <RearAxle  maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8"
                     positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="Ego">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <WorldPosition x="{init_x}" y="{init_y}" z="{init_z}" h="0.0"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
        </Private>
      </Actions>
    </Init>
    <Story name="MainStory">
      <Act name="EgoAct">
        <ManeuverGroup name="EgoGroup" maximumExecutionCount="1">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="Ego"/>
          </Actors>
          <Maneuver name="EgoManeuver">
            <Event name="EgoEvent" priority="override">
              <Action name="EgoAction">
                <PrivateAction>
                  <RoutingAction>
                    <FollowTrajectoryAction>
                      <TrajectoryRef>
                        <Trajectory closed="false" name="ego_trajectory">
                          <Shape>
                            <Polyline>
{vertex_xml}
                            </Polyline>
                          </Shape>
                        </Trajectory>
                      </TrajectoryRef>
                      <TimeReference>
                        <Timing domainAbsoluteRelative="relative" offset="0.0" scale="1.0"/>
                      </TimeReference>
                      <TrajectoryFollowingMode followingMode="position"/>
                    </FollowTrajectoryAction>
                  </RoutingAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="Start" delay="0" conditionEdge="none">
                    <ByValueCondition>
                      <SimulationTimeCondition value="0.0" rule="greaterOrEqual"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <StartTrigger>
          <ConditionGroup>
            <Condition name="ActStart" delay="0" conditionEdge="none">
              <ByValueCondition>
                <SimulationTimeCondition value="0.0" rule="greaterOrEqual"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
      </Act>
    </Story>
    <StopTrigger>
      <ConditionGroup>
        <Condition name="End" delay="0" conditionEdge="none">
          <ByValueCondition>
            <SimulationTimeCondition value="{end_time}" rule="greaterOrEqual"/>
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>
""")


def _run_scenario(tmp_path: Path, xosc_text: str, step_size: float = 0.05):
    scenario_file = tmp_path / "test.xosc"
    scenario_file.write_text(xosc_text)
    scenario = ScenarioParser().parse(scenario_file)
    engine = SimulationEngine(scenario, step_size=step_size)
    return list(engine.run()), scenario


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parser_reads_follow_trajectory_action(tmp_path: Path) -> None:
    """FollowTrajectoryAction with Polyline vertices should be parsed correctly."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (2.0, 20.0, 0.0, 0.0)]
    xosc = _make_xosc(vertices)
    scenario_file = tmp_path / "traj.xosc"
    scenario_file.write_text(xosc)

    scenario = ScenarioParser().parse(scenario_file)
    story = scenario.stories[0]
    event = story.acts[0].maneuver_groups[0].maneuvers[0].events[0]
    assert len(event.actions) == 1
    action = event.actions[0]
    assert isinstance(action, FollowTrajectoryAction)
    assert len(action.vertices) == 2
    assert action.vertices[0].time == pytest.approx(0.0)
    assert action.vertices[0].position.x == pytest.approx(0.0)
    assert action.vertices[1].time == pytest.approx(2.0)
    assert action.vertices[1].position.x == pytest.approx(20.0)


def test_parser_trajectory_vertex_dataclass() -> None:
    """TrajectoryVertex fields are accessible."""
    v = TrajectoryVertex(time=1.5, position=None)  # type: ignore[arg-type]
    assert v.time == 1.5


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

def test_trajectory_start_position(tmp_path: Path) -> None:
    """Entity must be at the first vertex at t=0."""
    vertices = [(0.0, 10.0, 5.0, 0.0), (5.0, 60.0, 5.0, 0.0)]
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)

    t0, gt0 = frames[0]
    assert t0 == pytest.approx(0.0)
    ego = gt0.moving_object[0]
    assert ego.base.position.x == pytest.approx(10.0, abs=0.01)
    assert ego.base.position.y == pytest.approx(5.0, abs=0.01)


def test_trajectory_end_position_clamped(tmp_path: Path) -> None:
    """After the last vertex the entity should be frozen at the final waypoint."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (1.0, 10.0, 0.0, 0.0)]
    # scenario runs until t ≥ 1.1, so the last frame is after the trajectory ends
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)

    last_t, last_gt = frames[-1]
    assert last_t >= 1.0
    ego = last_gt.moving_object[0]
    assert ego.base.position.x == pytest.approx(10.0, abs=0.01)


def test_trajectory_midpoint_interpolation(tmp_path: Path) -> None:
    """Entity should be at the midpoint at t halfway between two vertices."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (4.0, 40.0, 0.0, 0.0)]
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)

    # Find the frame closest to t=2.0 (midpoint)
    mid_frame = min(frames, key=lambda tf: abs(tf[0] - 2.0))
    t_mid, gt_mid = mid_frame
    ego = gt_mid.moving_object[0]
    # At t=2.0, x should be ~20.0 (half of 40)
    assert ego.base.position.x == pytest.approx(20.0, abs=0.6)


def test_trajectory_heading_derived_from_direction(tmp_path: Path) -> None:
    """Heading should be derived from the polyline direction (East = 0 rad)."""
    # Pure eastward trajectory
    vertices = [(0.0, 0.0, 0.0, 0.0), (2.0, 20.0, 0.0, 0.0)]
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)

    # At any mid-segment frame, heading should be ~0 rad (East)
    _, gt = frames[10]
    ego = gt.moving_object[0]
    assert ego.base.orientation.yaw == pytest.approx(0.0, abs=0.01)


def test_trajectory_diagonal_heading(tmp_path: Path) -> None:
    """Diagonal segment heading should be 45° (π/4 rad)."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (2.0, 10.0, 10.0, 0.0)]
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)

    _, gt = frames[5]
    ego = gt.moving_object[0]
    assert ego.base.orientation.yaw == pytest.approx(math.pi / 4, abs=0.01)


def test_trajectory_odometer_accumulates(tmp_path: Path) -> None:
    """Odometer should accumulate the distance travelled along the trajectory."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (1.0, 10.0, 0.0, 0.0)]
    scenario_file = tmp_path / "traj.xosc"
    scenario_file.write_text(_make_xosc(vertices))
    scenario = ScenarioParser().parse(scenario_file)
    engine = SimulationEngine(scenario, step_size=0.05)
    list(engine.run())

    # After trajectory completes, odometer should be ~10 m
    entity = engine._entities["Ego"]
    assert entity.odometer == pytest.approx(10.0, abs=0.5)


def test_trajectory_frame_timestamps_monotonic(tmp_path: Path) -> None:
    """Timestamps must be strictly non-decreasing."""
    vertices = [(0.0, 0.0, 0.0, 0.0), (3.0, 30.0, 0.0, 0.0)]
    frames, _ = _run_scenario(tmp_path, _make_xosc(vertices), step_size=0.05)
    timestamps = [t for t, _ in frames]
    assert timestamps == sorted(timestamps)
