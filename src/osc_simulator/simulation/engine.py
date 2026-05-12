"""Kinematic simulation engine.

Runs the scenario forward in fixed time steps, evaluates OpenSCENARIO
conditions, applies actions, and yields (timestamp, GroundTruth) pairs.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import osi3.osi_groundtruth_pb2 as osi_gt
import osi3.osi_object_pb2 as osi_object

from osc_simulator.parser.openscenario import (
    Act,
    Condition,
    FollowTrajectoryAction,
    LaneChangeAction,
    Maneuver,
    ManeuverGroup,
    Scenario,
    SpeedAction,
    Story,
    TeleportAction,
)
from osc_simulator.simulation.entities import EntityRuntimeState

_NANO = 1_000_000_000  # seconds → nanoseconds


def _compare(actual: float, rule: str, threshold: float) -> bool:
    return {
        "greaterThan": actual > threshold,
        "greaterOrEqual": actual >= threshold,
        "lessThan": actual < threshold,
        "lessOrEqual": actual <= threshold,
        "equalTo": math.isclose(actual, threshold, abs_tol=1e-6),
    }.get(rule, False)


class _EventState:
    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.trigger_time: float | None = None
        self.action_states: list[_ActionExecutionState] = []


class _ActionExecutionState:
    def __init__(
        self,
        action: SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction,
        actor_names: list[str],
    ) -> None:
        self.action = action
        self.actor_names = actor_names
        self.started = False
        self.completed = False


class SimulationEngine:
    """Execute a :class:`~osc_simulator.parser.openscenario.Scenario`."""

    def __init__(self, scenario: Scenario, step_size: float = 0.05) -> None:
        self._scenario = scenario
        self._dt = step_size
        self._time = 0.0

        # Build runtime entity states
        self._entities: dict[str, EntityRuntimeState] = {
            e.name: EntityRuntimeState.from_definition(e) for e in scenario.entities
        }

        # Apply any extra init actions (e.g. FollowTrajectoryAction defined in Init)
        for entity_name, actions in scenario.init_actions.items():
            entity = self._entities.get(entity_name)
            if entity is None:
                continue
            for action in actions:
                if isinstance(action, FollowTrajectoryAction):
                    entity.apply_trajectory(action.vertices)

        # Track which events have already fired
        self._event_states: dict[str, _EventState] = {}

        # Track storyboard element lifecycle (started/completed) for
        # StoryboardElementStateCondition support.
        self._started_elements: set[str] = set()
        self._completed_elements: set[str] = set()
        self._storyboard_lookup = self._build_storyboard_lookup()

    # ------------------------------------------------------------------
    # Public

    def run(self) -> Iterator[tuple[float, Any]]:
        """Yield ``(timestamp_seconds, osi3.GroundTruth)`` for each time step."""
        while not self._stop_triggered():
            self._evaluate_storyboard()
            gt = self._build_ground_truth()
            yield self._time, gt
            self._advance()

        # Emit final frame at stop time
        yield self._time, self._build_ground_truth()

    # ------------------------------------------------------------------
    # Condition evaluation

    def _stop_triggered(self) -> bool:
        return self._conditions_met(self._scenario.stop_conditions)

    def _conditions_met(self, conditions: list[Condition]) -> bool:
        if not conditions:
            return False
        return any(self._condition_true(c) for c in conditions)

    def _condition_true(self, cond: Condition) -> bool:
        p = cond.params
        kind = p.get("type")
        if kind == "simulation_time":
            return _compare(self._time, p["rule"], p["value"])
        if kind == "distance":
            entity_ref = p.get("entity_ref", "")
            triggering = p.get("triggering_entities", list(self._entities.keys()))
            coord_sys = p.get("coordinate_system", "entity")
            rel_dist_type = p.get("relative_distance_type", "euclidianDistance")
            for ename in triggering:
                e1 = self._entities.get(ename)
                e2 = self._entities.get(entity_ref)
                if e1 and e2:
                    if coord_sys == "entity" and rel_dist_type == "longitudinal":
                        # Project separation onto e1's heading direction
                        dx = e2.x - e1.x
                        dy = e2.y - e1.y
                        d = abs(dx * math.cos(e1.heading) + dy * math.sin(e1.heading))
                    else:
                        d = math.hypot(e1.x - e2.x, e1.y - e2.y)
                    if _compare(d, p["rule"], p["value"]):
                        return True
        if kind == "ttc":
            # Simplified TTC: longitudinal closing speed
            triggering = p.get("triggering_entities", [])
            for ename in triggering:
                e1 = self._entities.get(ename)
                if e1 and e1.speed > 0:
                    ttc = p["value"]  # placeholder
                    if _compare(ttc, p["rule"], p["value"]):
                        return True
        if kind == "speed":
            triggering = p.get("triggering_entities", list(self._entities.keys()))
            for ename in triggering:
                e1 = self._entities.get(ename)
                if e1 and _compare(e1.speed, p["rule"], p["value"]):
                    return True
        if kind == "traveled_distance":
            triggering = p.get("triggering_entities", list(self._entities.keys()))
            for ename in triggering:
                e1 = self._entities.get(ename)
                if e1 and e1.odometer >= p["value"]:
                    return True
        if kind == "storyboard_element_state":
            element_type = p.get("storyboard_element_type", "").lower()
            element_ref = p.get("storyboard_element_ref", "")
            requested_state = p.get("state", "")
            key = self._storyboard_lookup.get((element_type, element_ref), element_ref)
            started = key in self._started_elements
            completed = key in self._completed_elements
            return self._match_storyboard_state(started, completed, requested_state)
        return False

    # ------------------------------------------------------------------
    # Action dispatch

    def _apply_action(
        self,
        action: SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction,
        actor_names: list[str],
    ) -> None:
        for name in actor_names:
            entity = self._entities.get(name)
            if entity is None:
                continue
            if isinstance(action, SpeedAction):
                entity.apply_speed_action(
                    action.target_speed,
                    shape=action.dynamics_shape,
                    duration=action.dynamics_value,
                )
            elif isinstance(action, LaneChangeAction):
                entity.apply_lateral_action(
                    action.target_lane_offset,
                    shape=action.dynamics_shape,
                    duration=action.dynamics_value,
                )
            elif isinstance(action, TeleportAction):
                entity.apply_teleport(action.position)
            elif isinstance(action, FollowTrajectoryAction):
                entity.apply_trajectory(action.vertices)

    def _is_action_completed(
        self,
        action: SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction,
        actor_names: list[str],
    ) -> bool:
        if isinstance(action, TeleportAction):
            return True

        if isinstance(action, SpeedAction):
            if action.dynamics_shape == "step" or action.dynamics_value <= 0.0:
                return True
            for name in actor_names:
                entity = self._entities.get(name)
                if entity is None:
                    continue
                if entity.dynamics is not None:
                    return False
            return True

        if isinstance(action, LaneChangeAction):
            if action.dynamics_value <= 0.0:
                return True
            for name in actor_names:
                entity = self._entities.get(name)
                if entity is None:
                    continue
                if entity.lateral is not None:
                    return False
            return True

        if isinstance(action, FollowTrajectoryAction):
            if not action.vertices:
                return True
            end_time = action.vertices[-1].time
            for name in actor_names:
                entity = self._entities.get(name)
                if entity is None:
                    continue
                if entity.trajectory is None:
                    return True
                if entity._trajectory_time < end_time:
                    return False
            return True

        return False

    # ------------------------------------------------------------------
    # Storyboard execution

    def _advance(self) -> None:
        for entity in self._entities.values():
            entity.step(self._dt)
        self._time += self._dt
        # Clamp floating-point drift
        self._time = round(self._time, 9)

    def _evaluate_storyboard(self) -> None:
        self._update_event_action_completion_states()
        for story in self._scenario.stories:
            self._evaluate_story(story)
        self._update_event_action_completion_states()
        self._update_storyboard_completion_states()

    def _evaluate_story(self, story: Story) -> None:
        story_key = self._story_key(story)
        self._started_elements.add(story_key)
        for act in story.acts:
            self._evaluate_act(story, act)

    def _evaluate_act(self, story: Story, act: Act) -> None:
        act_key = self._act_key(story, act)
        if act.has_start_trigger and not self._conditions_met(act.start_conditions):
            return
        self._started_elements.add(act_key)
        for mg in act.maneuver_groups:
            self._evaluate_maneuver_group(story, act, mg)

    def _evaluate_maneuver_group(self, story: Story, act: Act, mg: ManeuverGroup) -> None:
        mg_key = self._maneuver_group_key(story, act, mg)
        self._started_elements.add(mg_key)
        for maneuver in mg.maneuvers:
            self._evaluate_maneuver(story, act, mg, maneuver, mg.actors)

    def _evaluate_maneuver(
        self,
        story: Story,
        act: Act,
        mg: ManeuverGroup,
        maneuver: Maneuver,
        actors: list[str],
    ) -> None:
        maneuver_key = self._maneuver_key(story, act, mg, maneuver)
        self._started_elements.add(maneuver_key)
        for event in maneuver.events:
            event_key = self._event_key(story, act, mg, maneuver, event)
            state = self._event_states.setdefault(event_key, _EventState())
            if state.started and event.priority != "parallel":
                continue
            if not event.has_start_trigger or self._conditions_met(event.start_conditions):
                # Enforce condition delay: record when conditions first became true
                delay = max((c.delay for c in event.start_conditions), default=0.0)
                if delay > 0.0:
                    if state.trigger_time is None:
                        state.trigger_time = self._time
                    if self._time - state.trigger_time < delay:
                        continue
                state.started = True
                self._started_elements.add(event_key)
                for action in event.actions:
                    action_state = _ActionExecutionState(action, list(actors))
                    state.action_states.append(action_state)
                    action_state.started = True
                    self._apply_action(action, actors)
                    action_state.completed = self._is_action_completed(action, actors)
                if not state.action_states or all(a.completed for a in state.action_states):
                    state.completed = True
                    self._completed_elements.add(event_key)
            else:
                # Reset trigger time if conditions no longer hold (rising-edge semantics)
                state.trigger_time = None

    def _update_event_action_completion_states(self) -> None:
        for state in self._event_states.values():
            if not state.started or state.completed:
                continue
            for action_state in state.action_states:
                if action_state.completed:
                    continue
                action_state.completed = self._is_action_completed(
                    action_state.action, action_state.actor_names
                )
            if all(a.completed for a in state.action_states):
                state.completed = True

    def _story_key(self, story: Story) -> str:
        return f"story:{story.name}"

    def _act_key(self, story: Story, act: Act) -> str:
        return f"act:{story.name}/{act.name}"

    def _maneuver_group_key(self, story: Story, act: Act, mg: ManeuverGroup) -> str:
        return f"maneuvergroup:{story.name}/{act.name}/{mg.name}"

    def _maneuver_key(self, story: Story, act: Act, mg: ManeuverGroup, maneuver: Maneuver) -> str:
        return f"maneuver:{story.name}/{act.name}/{mg.name}/{maneuver.name}"

    def _event_key(
        self, story: Story, act: Act, mg: ManeuverGroup, maneuver: Maneuver, event: Any
    ) -> str:
        return f"event:{story.name}/{act.name}/{mg.name}/{maneuver.name}/{event.name}"

    def _build_storyboard_lookup(self) -> dict[tuple[str, str], str]:
        lookup: dict[tuple[str, str], str] = {}
        for story in self._scenario.stories:
            story_key = self._story_key(story)
            lookup.setdefault(("story", story.name), story_key)
            for act in story.acts:
                act_key = self._act_key(story, act)
                lookup.setdefault(("act", act.name), act_key)
                for mg in act.maneuver_groups:
                    mg_key = self._maneuver_group_key(story, act, mg)
                    lookup.setdefault(("maneuvergroup", mg.name), mg_key)
                    for maneuver in mg.maneuvers:
                        maneuver_key = self._maneuver_key(story, act, mg, maneuver)
                        lookup.setdefault(("maneuver", maneuver.name), maneuver_key)
                        for event in maneuver.events:
                            event_key = self._event_key(story, act, mg, maneuver, event)
                            lookup.setdefault(("event", event.name), event_key)
        return lookup

    def _update_storyboard_completion_states(self) -> None:
        for story in self._scenario.stories:
            story_key = self._story_key(story)
            acts_complete = True
            for act in story.acts:
                act_key = self._act_key(story, act)
                mgs_complete = True
                for mg in act.maneuver_groups:
                    mg_key = self._maneuver_group_key(story, act, mg)
                    maneuvers_complete = True
                    for maneuver in mg.maneuvers:
                        maneuver_key = self._maneuver_key(story, act, mg, maneuver)
                        events_complete = True
                        for event in maneuver.events:
                            event_key = self._event_key(story, act, mg, maneuver, event)
                            event_state = self._event_states.get(event_key)
                            if event_state is not None and event_state.completed:
                                self._completed_elements.add(event_key)
                            else:
                                events_complete = False
                        if events_complete and maneuver.events:
                            self._completed_elements.add(maneuver_key)
                        else:
                            maneuvers_complete = False
                    if maneuvers_complete and mg.maneuvers:
                        self._completed_elements.add(mg_key)
                    else:
                        mgs_complete = False
                if mgs_complete and act.maneuver_groups:
                    self._completed_elements.add(act_key)
                else:
                    acts_complete = False
            if acts_complete and story.acts:
                self._completed_elements.add(story_key)

    def _match_storyboard_state(self, started: bool, completed: bool, state: str) -> bool:
        if state == "standbyState":
            return not started
        if state in {"runningState", "startTransition"}:
            return started and not completed
        if state in {"completeState", "endTransition"}:
            return completed
        if state in {"stopTransition", "skipTransition"}:
            return False
        return False

    # ------------------------------------------------------------------
    # OSI GroundTruth construction

    def _build_ground_truth(self) -> Any:
        gt = osi_gt.GroundTruth()
        ts = gt.timestamp
        whole_seconds = int(self._time)
        nanos = round((self._time - whole_seconds) * _NANO)
        ts.seconds = whole_seconds
        ts.nanos = nanos

        gt.country_code = 276  # Germany
        gt.proj_string = "EPSG:4326"  # WGS 84
        gt.map_reference = "none"
        gt.proj_frame_offset.position.x = 0.0
        gt.proj_frame_offset.position.y = 0.0
        gt.proj_frame_offset.position.z = 0.0
        gt.proj_frame_offset.yaw = 0.0

        for name, state in self._entities.items():
            defn = state.definition
            if defn.category in ("car", "truck", "van", "bus", "motorcycle", "trailer"):
                mv = gt.moving_object.add()
                self._fill_moving_object(mv, state)
            elif defn.category == "pedestrian":
                mv = gt.moving_object.add()
                mv.type = 3  # TYPE_PEDESTRIAN
                self._fill_moving_object(mv, state)
            else:
                so = gt.stationary_object.add()
                so.id.value = abs(hash(name)) % (2**32)
                so.base.position.x = state.x
                so.base.position.y = state.y
                so.base.position.z = state.z

        if len(gt.moving_object) > 0:
            gt.host_vehicle_id.CopyFrom(gt.moving_object[0].id)

        return gt

    def _fill_moving_object(self, mv: Any, state: EntityRuntimeState) -> None:
        defn = state.definition
        mv.id.value = abs(hash(defn.name)) % (2**32)

        # Type
        cat = defn.category
        if cat in ("car", "truck", "van", "bus", "motorcycle", "trailer"):
            mv.type = 2  # TYPE_VEHICLE
            mv.vehicle_attributes.number_wheels = (
                4 if cat in ("car", "van") else 6 if cat in ("truck", "bus") else 2
            )
            mv.vehicle_attributes.bbcenter_to_rear.x = defn.bounding_box[0] / 2
            mv.vehicle_classification.type = (
                osi_object.MovingObject.VehicleClassification.TYPE_CAR
                if cat == "car"
                else osi_object.MovingObject.VehicleClassification.TYPE_VAN
                if cat == "van"
                else osi_object.MovingObject.VehicleClassification.TYPE_HEAVY_TRUCK
                if cat == "truck"
                else osi_object.MovingObject.VehicleClassification.TYPE_BUS
                if cat == "bus"
                else osi_object.MovingObject.VehicleClassification.TYPE_TRAILER
                if cat == "trailer"
                else osi_object.MovingObject.VehicleClassification.TYPE_MOTORCYCLE
            )
            mv.vehicle_classification.role = (
                osi_object.MovingObject.VehicleClassification.ROLE_CIVIL
            )
        elif cat == "pedestrian":
            mv.type = 3  # TYPE_PEDESTRIAN

        # Base
        mv.base.position.x = state.x
        mv.base.position.y = state.y
        mv.base.position.z = state.z

        # Orientation (yaw = heading)
        mv.base.orientation.yaw = state.heading
        mv.base.orientation.pitch = 0.0
        mv.base.orientation.roll = 0.0

        # Velocity
        mv.base.velocity.x = state.speed * math.cos(state.heading)
        mv.base.velocity.y = state.speed * math.sin(state.heading)
        mv.base.velocity.z = 0.0

        # Acceleration
        mv.base.acceleration.x = state.acceleration * math.cos(state.heading)
        mv.base.acceleration.y = state.acceleration * math.sin(state.heading)
        mv.base.acceleration.z = 0.0

        # Bounding box
        length, width, height = defn.bounding_box
        mv.base.dimension.length = length
        mv.base.dimension.width = width
        mv.base.dimension.height = height
