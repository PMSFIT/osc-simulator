"""Tests for the OpenSCENARIO parser."""

from pathlib import Path

import pytest

from osc_simulator.parser.openscenario import ScenarioParser, SpeedAction

EXAMPLE = Path(__file__).parent.parent / "examples" / "simple_scenario.xosc"


def test_parse_entities() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    assert len(scenario.entities) == 2
    names = {e.name for e in scenario.entities}
    assert names == {"Ego", "NPC"}


def test_parse_initial_positions() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    ego = next(e for e in scenario.entities if e.name == "Ego")
    npc = next(e for e in scenario.entities if e.name == "NPC")
    assert ego.initial_state.position.x == pytest.approx(0.0)
    assert npc.initial_state.position.x == pytest.approx(50.0)


def test_parse_initial_speeds() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    ego = next(e for e in scenario.entities if e.name == "Ego")
    npc = next(e for e in scenario.entities if e.name == "NPC")
    assert ego.initial_state.speed == pytest.approx(20.0)
    assert npc.initial_state.speed == pytest.approx(10.0)


def test_parse_stop_trigger() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    assert len(scenario.stop_conditions) == 1
    cond = scenario.stop_conditions[0]
    assert cond.params["type"] == "simulation_time"
    assert cond.params["value"] == pytest.approx(10.0)


def test_parse_story_actions() -> None:
    scenario = ScenarioParser().parse(EXAMPLE)
    assert len(scenario.stories) == 1
    story = scenario.stories[0]
    assert len(story.acts) == 1
    act = story.acts[0]
    assert len(act.maneuver_groups) == 1
    mg = act.maneuver_groups[0]
    assert "NPC" in mg.actors
    event = mg.maneuvers[0].events[0]
    assert len(event.actions) == 1
    action = event.actions[0]
    assert isinstance(action, SpeedAction)
    assert action.target_speed == pytest.approx(20.0)
    assert action.dynamics_shape == "linear"
