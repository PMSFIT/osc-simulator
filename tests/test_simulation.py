"""Integration tests: parse → simulate → verify kinematics."""

import textwrap
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


def test_storyboard_element_state_condition_triggers_dependent_event(tmp_path: Path) -> None:
        xosc = textwrap.dedent(
                """<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
    <FileHeader revMajor="1" revMinor="1" date="2024-01-01T00:00:00"
                            description="State condition integration test" author="test"/>
    <RoadNetwork><LogicFile filepath=""/><SceneGraphFile filepath=""/></RoadNetwork>
    <Entities>
        <ScenarioObject name="Ego"><Vehicle name="Car" vehicleCategory="car"/></ScenarioObject>
    </Entities>
    <Storyboard>
        <Init>
            <Actions>
                <Private entityRef="Ego">
                    <PrivateAction>
                        <TeleportAction>
                            <Position><WorldPosition x="0.0" y="0.0" z="0.0" h="0.0"/></Position>
                        </TeleportAction>
                    </PrivateAction>
                </Private>
            </Actions>
        </Init>
        <Story name="MainStory">
            <Act name="MainAct">
                <ManeuverGroup name="MainGroup">
                    <Actors><EntityRef entityRef="Ego"/></Actors>
                    <Maneuver name="MainManeuver">
                        <Event name="E1" priority="overwrite">
                            <Action name="SetLowSpeed">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="5.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="StartE1" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <SimulationTimeCondition value="0.0" rule="greaterOrEqual"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                        <Event name="E2" priority="overwrite">
                            <Action name="SetHighSpeed">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="12.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="AfterE1" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <StoryboardElementStateCondition storyboardElementRef="E1"
                                                storyboardElementType="event" state="completeState"/>
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
                        <SimulationTimeCondition value="0.2" rule="greaterOrEqual"/>
                    </ByValueCondition>
                </Condition>
            </ConditionGroup>
        </StopTrigger>
    </Storyboard>
</OpenSCENARIO>
"""
        )
        scenario_file = tmp_path / "state_condition.xosc"
        scenario_file.write_text(xosc)

        scenario = ScenarioParser().parse(scenario_file)
        engine = SimulationEngine(scenario, step_size=0.05)
        frames = list(engine.run())

        _, gt0 = frames[0]
        ego = gt0.moving_object[0]
        speed = (ego.base.velocity.x**2 + ego.base.velocity.y**2) ** 0.5
        assert speed == pytest.approx(12.0, abs=0.2)


def test_storyboard_running_state_precedes_complete_state(tmp_path: Path) -> None:
        xosc = textwrap.dedent(
                """<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
    <FileHeader revMajor="1" revMinor="1" date="2024-01-01T00:00:00"
                            description="State sequencing test" author="test"/>
    <RoadNetwork><LogicFile filepath=""/><SceneGraphFile filepath=""/></RoadNetwork>
    <Entities>
        <ScenarioObject name="Ego"><Vehicle name="Car" vehicleCategory="car"/></ScenarioObject>
    </Entities>
    <Storyboard>
        <Init>
            <Actions>
                <Private entityRef="Ego">
                    <PrivateAction>
                        <TeleportAction>
                            <Position><WorldPosition x="0.0" y="0.0" z="0.0" h="0.0"/></Position>
                        </TeleportAction>
                    </PrivateAction>
                </Private>
            </Actions>
        </Init>
        <Story name="MainStory">
            <Act name="MainAct">
                <ManeuverGroup name="MainGroup">
                    <Actors><EntityRef entityRef="Ego"/></Actors>
                    <Maneuver name="MainManeuver">
                        <Event name="E1" priority="overwrite">
                            <Action name="RampSpeed">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionDynamics dynamicsShape="linear" value="1.0" dynamicsDimension="time"/>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="10.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="StartE1" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <SimulationTimeCondition value="0.0" rule="greaterOrEqual"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                        <Event name="E2" priority="overwrite">
                            <Action name="OnRunning">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="7.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="E1Running" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <StoryboardElementStateCondition storyboardElementRef="E1"
                                                storyboardElementType="event" state="runningState"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                        <Event name="E3" priority="overwrite">
                            <Action name="OnComplete">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="12.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="E1Complete" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <StoryboardElementStateCondition storyboardElementRef="E1"
                                                storyboardElementType="event" state="completeState"/>
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
                        <SimulationTimeCondition value="0.2" rule="greaterOrEqual"/>
                    </ByValueCondition>
                </Condition>
            </ConditionGroup>
        </StopTrigger>
    </Storyboard>
</OpenSCENARIO>
"""
        )
        scenario_file = tmp_path / "state_sequence.xosc"
        scenario_file.write_text(xosc)

        scenario = ScenarioParser().parse(scenario_file)
        engine = SimulationEngine(scenario, step_size=0.05)
        frames = list(engine.run())

        # E2 should trigger from E1 runningState, setting speed to 7 m/s.
        # E3 (completeState) must not trigger in the same tick.
        _, gt0 = frames[0]
        ego = gt0.moving_object[0]
        speed0 = (ego.base.velocity.x**2 + ego.base.velocity.y**2) ** 0.5
        assert speed0 == pytest.approx(7.0, abs=0.2)


def test_storyboard_complete_state_triggers_on_later_frame(tmp_path: Path) -> None:
        xosc = textwrap.dedent(
                """<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
    <FileHeader revMajor="1" revMinor="1" date="2024-01-01T00:00:00"
                            description="Complete state timing test" author="test"/>
    <RoadNetwork><LogicFile filepath=""/><SceneGraphFile filepath=""/></RoadNetwork>
    <Entities>
        <ScenarioObject name="Ego"><Vehicle name="Car" vehicleCategory="car"/></ScenarioObject>
    </Entities>
    <Storyboard>
        <Init>
            <Actions>
                <Private entityRef="Ego">
                    <PrivateAction>
                        <TeleportAction>
                            <Position><WorldPosition x="0.0" y="0.0" z="0.0" h="0.0"/></Position>
                        </TeleportAction>
                    </PrivateAction>
                </Private>
            </Actions>
        </Init>
        <Story name="MainStory">
            <Act name="MainAct">
                <ManeuverGroup name="MainGroup">
                    <Actors><EntityRef entityRef="Ego"/></Actors>
                    <Maneuver name="MainManeuver">
                        <Event name="E1" priority="overwrite">
                            <Action name="RampSpeed">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionDynamics dynamicsShape="linear" value="0.5" dynamicsDimension="time"/>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="10.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="StartE1" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <SimulationTimeCondition value="0.0" rule="greaterOrEqual"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                        <Event name="E2" priority="overwrite">
                            <Action name="OnRunning">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="6.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="E1Running" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <StoryboardElementStateCondition storyboardElementRef="E1"
                                                storyboardElementType="event" state="runningState"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                        <Event name="E3" priority="overwrite">
                            <Action name="OnComplete">
                                <PrivateAction>
                                    <LongitudinalAction>
                                        <SpeedAction>
                                            <SpeedActionTarget><AbsoluteTargetSpeed value="11.0"/></SpeedActionTarget>
                                        </SpeedAction>
                                    </LongitudinalAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="E1Complete" delay="0" conditionEdge="none">
                                        <ByValueCondition>
                                            <StoryboardElementStateCondition storyboardElementRef="E1"
                                                storyboardElementType="event" state="completeState"/>
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
                        <SimulationTimeCondition value="0.1" rule="greaterOrEqual"/>
                    </ByValueCondition>
                </Condition>
            </ConditionGroup>
        </StopTrigger>
    </Storyboard>
</OpenSCENARIO>
"""
        )
        scenario_file = tmp_path / "state_complete_timing.xosc"
        scenario_file.write_text(xosc)

        scenario = ScenarioParser().parse(scenario_file)
        engine = SimulationEngine(scenario, step_size=0.05)
        frames = list(engine.run())

        # Frame 0: E1 has started but is not complete yet, so E2 (runningState) can fire.
        _, gt0 = frames[0]
        speed0 = (gt0.moving_object[0].base.velocity.x**2 + gt0.moving_object[0].base.velocity.y**2) ** 0.5
        assert speed0 == pytest.approx(6.0, abs=0.2)

        # Frame 1: E1 completion is observed on the next evaluation cycle, so E3 can fire then.
        _, gt1 = frames[1]
        speed1 = (gt1.moving_object[0].base.velocity.x**2 + gt1.moving_object[0].base.velocity.y**2) ** 0.5
        assert speed1 == pytest.approx(11.0, abs=0.2)
