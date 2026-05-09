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
        self.fired = False
        self.trigger_time: float | None = None


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

    # ------------------------------------------------------------------
    # Public

    def run(self) -> Iterator[tuple[float, Any]]:
        """Yield ``(timestamp_seconds, osi3.GroundTruth)`` for each time step."""
        while not self._stop_triggered():
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

    # ------------------------------------------------------------------
    # Storyboard execution

    def _advance(self) -> None:
        self._evaluate_storyboard()
        for entity in self._entities.values():
            entity.step(self._dt)
        self._time += self._dt
        # Clamp floating-point drift
        self._time = round(self._time, 9)

    def _evaluate_storyboard(self) -> None:
        for story in self._scenario.stories:
            self._evaluate_story(story)

    def _evaluate_story(self, story: Story) -> None:
        for act in story.acts:
            self._evaluate_act(act)

    def _evaluate_act(self, act: Act) -> None:
        if act.has_start_trigger and not self._conditions_met(act.start_conditions):
            return
        for mg in act.maneuver_groups:
            self._evaluate_maneuver_group(mg)

    def _evaluate_maneuver_group(self, mg: ManeuverGroup) -> None:
        for maneuver in mg.maneuvers:
            self._evaluate_maneuver(maneuver, mg.actors)

    def _evaluate_maneuver(self, maneuver: Maneuver, actors: list[str]) -> None:
        for event in maneuver.events:
            event_key = f"{maneuver.name}/{event.name}"
            state = self._event_states.setdefault(event_key, _EventState())
            if state.fired and event.priority != "parallel":
                continue
            if not event.has_start_trigger or self._conditions_met(event.start_conditions):
                # Enforce condition delay: record when conditions first became true
                delay = max((c.delay for c in event.start_conditions), default=0.0)
                if delay > 0.0:
                    if state.trigger_time is None:
                        state.trigger_time = self._time
                    if self._time - state.trigger_time < delay:
                        continue
                for action in event.actions:
                    self._apply_action(action, actors)
                state.fired = True
            else:
                # Reset trigger time if conditions no longer hold (rising-edge semantics)
                state.trigger_time = None

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
