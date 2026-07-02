"""ASAM OpenSCENARIO 1.x XML parser.

Supports the subset of the schema needed to drive the built-in kinematic
simulation engine:
  - Entities (Vehicle / Pedestrian / MiscObject)
  - Init PrivateActions: TeleportAction (WorldPosition) and SpeedAction
  - Storyboard: Story → Act → ManeuverGroup → Maneuver → Event
  - Actions: LaneChangeAction, SpeedAction, TeleportAction, FollowTrajectoryAction
  - Conditions: SimulationTimeCondition, EntityCondition (distance/TTC/speed/
    traveledDistance), ByValueCondition (SimulationTime, Parameter)
  - StopTrigger via SimulationTimeCondition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import QName

from lxml import etree

# ---------------------------------------------------------------------------
# Domain model (parse result)
# ---------------------------------------------------------------------------


@dataclass
class WorldPosition:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    h: float = 0.0  # heading (rad)
    p: float = 0.0  # pitch (rad)
    r: float = 0.0  # roll  (rad)


@dataclass
class EntityState:
    position: WorldPosition = field(default_factory=WorldPosition)
    speed: float = 0.0  # m/s longitudinal


@dataclass
class EntityDef:
    name: str
    category: str  # "car", "truck", "pedestrian", …
    bounding_box: tuple[float, float, float] = (4.5, 2.0, 1.5)  # l, w, h metres
    bounding_box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_state: EntityState = field(default_factory=EntityState)


@dataclass
class SpeedAction:
    target_speed: float  # m/s
    dynamics_shape: str = "step"  # "step", "linear", "sinusoidal"
    dynamics_value: float = 0.0  # time (s) or distance (m) for non-step


@dataclass
class LaneChangeAction:
    target_lane_offset: float  # metres lateral offset (simplified)
    dynamics_shape: str = "sinusoidal"
    dynamics_value: float = 3.0  # duration in seconds


@dataclass
class TeleportAction:
    position: WorldPosition


@dataclass
class TrajectoryVertex:
    """A single waypoint in a Polyline trajectory."""

    time: float  # absolute simulation time (seconds)
    position: WorldPosition


@dataclass
class FollowTrajectoryAction:
    """FollowTrajectoryAction with Polyline shape (osc-validation minimal subset)."""

    vertices: list[TrajectoryVertex] = field(default_factory=list)
    time_domain: str = "relative"  # "relative" | "absolute"


@dataclass
class Condition:
    name: str
    delay: float = 0.0
    # Populated by concrete subclass logic; kept as a dict for extensibility
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    name: str
    priority: str
    actions: list[SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction] = field(
        default_factory=list
    )
    start_conditions: list[Condition] = field(default_factory=list)
    has_start_trigger: bool = False  # whether a StartTrigger element was present


@dataclass
class Maneuver:
    name: str
    events: list[Event] = field(default_factory=list)


@dataclass
class ManeuverGroup:
    name: str
    actors: list[str] = field(default_factory=list)  # entity names
    maneuvers: list[Maneuver] = field(default_factory=list)


@dataclass
class Act:
    name: str
    maneuver_groups: list[ManeuverGroup] = field(default_factory=list)
    start_conditions: list[Condition] = field(default_factory=list)
    has_start_trigger: bool = False  # whether a StartTrigger element was present


@dataclass
class Story:
    name: str
    acts: list[Act] = field(default_factory=list)


@dataclass
class Scenario:
    description: str
    entities: list[EntityDef] = field(default_factory=list)
    stories: list[Story] = field(default_factory=list)
    stop_conditions: list[Condition] = field(default_factory=list)
    # Map: entity_name → list of init actions beyond position/speed
    init_actions: dict[
        str, list[SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction]
    ] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_DEG_TO_RAD = 3.141592653589793 / 180.0


def _float(el: etree._Element, attr: str, default: float = 0.0) -> float:
    v = el.get(attr)
    return float(v) if v is not None else default


def _str(el: etree._Element, attr: str, default: str = "") -> str:
    return el.get(attr, default)


class ScenarioParser:
    """Parse an OpenSCENARIO 1.x .xosc file into a :class:`Scenario`."""

    def parse(self, path: Path) -> Scenario:
        tree = etree.parse(str(path))
        root = tree.getroot()
        # Strip namespace if present
        tag = root.tag
        if isinstance(tag, QName):
            tag = tag.text
        assert isinstance(tag, str)
        ns = ""
        if tag.startswith("{"):
            ns = tag[: tag.index("}") + 1]

        def q(name: str) -> str:
            return f"{ns}{name}"

        header = root.find(q("FileHeader"))
        description = header.get("description", "") if header is not None else ""

        scenario = Scenario(description=description)

        self._parse_entities(root, q, scenario)
        self._parse_storyboard(root, q, scenario)

        return scenario

    # ------------------------------------------------------------------

    def _parse_entities(self, root: etree._Element, q: Any, scenario: Scenario) -> None:
        entities_el = root.find(q("Entities"))
        if entities_el is None:
            return
        for obj_el in entities_el.findall(q("ScenarioObject")):
            name = _str(obj_el, "name", "unknown")
            category, bbox, bbox_center = self._resolve_entity_type(obj_el, q)
            entity = EntityDef(
                name=name,
                category=category,
                bounding_box=bbox,
                bounding_box_center=bbox_center,
            )
            scenario.entities.append(entity)

    def _resolve_entity_type(
        self, obj_el: etree._Element, q: Any
    ) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
        for tag, cat in (
            (q("Vehicle"), "car"),
            (q("Pedestrian"), "pedestrian"),
            (q("MiscObject"), "object"),
        ):
            el = obj_el.find(tag)
            if el is not None:
                raw_cat = _str(el, "vehicleCategory") or _str(el, "pedestrianCategory") or cat
                bbox, bbox_center = self._parse_bounding_box(el, q)
                return raw_cat, bbox, bbox_center
        return "unknown", (4.5, 2.0, 1.5), (0.0, 0.0, 0.0)

    def _parse_bounding_box(
        self, el: etree._Element, q: Any
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        bb = el.find(q("BoundingBox"))
        if bb is None:
            return (4.5, 2.0, 1.5), (0.0, 0.0, 0.0)
        dim = bb.find(q("Dimensions"))
        center = bb.find(q("Center"))
        if dim is None:
            dimensions = (4.5, 2.0, 1.5)
        else:
            dimensions = (
                _float(dim, "length", 4.5),
                _float(dim, "width", 2.0),
                _float(dim, "height", 1.5),
            )
        center_xyz = (
            _float(center, "x", 0.0) if center is not None else 0.0,
            _float(center, "y", 0.0) if center is not None else 0.0,
            _float(center, "z", 0.0) if center is not None else 0.0,
        )
        return dimensions, center_xyz

    # ------------------------------------------------------------------

    def _parse_storyboard(self, root: etree._Element, q: Any, scenario: Scenario) -> None:
        sb = root.find(q("Storyboard"))
        if sb is None:
            return

        # Init actions
        init_el = sb.find(q("Init"))
        if init_el is not None:
            self._parse_init(init_el, q, scenario)

        # Stories
        for story_el in sb.findall(q("Story")):
            scenario.stories.append(self._parse_story(story_el, q))

        # StopTrigger
        stop_el = sb.find(q("StopTrigger"))
        if stop_el is not None:
            scenario.stop_conditions = self._parse_trigger(stop_el, q)

    def _parse_init(self, init_el: etree._Element, q: Any, scenario: Scenario) -> None:
        actions_el = init_el.find(q("Actions"))
        if actions_el is None:
            return
        for private_el in actions_el.findall(q("Private")):
            entity_name = _str(private_el, "entityRef")
            entity = next((e for e in scenario.entities if e.name == entity_name), None)
            extra: list[
                SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction
            ] = []
            for pa in private_el.findall(q("PrivateAction")):
                action = self._parse_private_action(pa, q)
                if action is None:
                    continue
                if isinstance(action, TeleportAction) and entity is not None:
                    entity.initial_state.position = action.position
                elif isinstance(action, SpeedAction) and entity is not None:
                    entity.initial_state.speed = action.target_speed
                else:
                    extra.append(action)
            if extra:
                scenario.init_actions.setdefault(entity_name, []).extend(extra)

        # Global Init actions (e.g. AddEntityAction)
        global_actions = list(actions_el.findall(q("GlobalAction")))
        for global_el in actions_el.findall(q("Global")):
            global_actions.extend(global_el.findall(q("GlobalAction")))

        for ga_el in global_actions:
            entity_action_el = ga_el.find(q("EntityAction"))
            if entity_action_el is None:
                continue
            entity_name = _str(entity_action_el, "entityRef")
            if not entity_name:
                continue
            entity = next((e for e in scenario.entities if e.name == entity_name), None)
            if entity is None:
                continue

            add_entity_el = entity_action_el.find(q("AddEntityAction"))
            if add_entity_el is not None:
                pos_el = add_entity_el.find(q("Position"))
                if pos_el is not None:
                    wp_el = pos_el.find(q("WorldPosition"))
                    if wp_el is not None:
                        entity.initial_state.position = self._parse_world_position(wp_el)

    def _parse_story(self, story_el: etree._Element, q: Any) -> Story:
        story = Story(name=_str(story_el, "name"))
        for act_el in story_el.findall(q("Act")):
            story.acts.append(self._parse_act(act_el, q))
        return story

    def _parse_act(self, act_el: etree._Element, q: Any) -> Act:
        act = Act(name=_str(act_el, "name"))
        start_el = act_el.find(q("StartTrigger"))
        if start_el is not None:
            act.start_conditions = self._parse_trigger(start_el, q)
            act.has_start_trigger = True
        for mg_el in act_el.findall(q("ManeuverGroup")):
            act.maneuver_groups.append(self._parse_maneuver_group(mg_el, q))
        return act

    def _parse_maneuver_group(self, mg_el: etree._Element, q: Any) -> ManeuverGroup:
        mg = ManeuverGroup(name=_str(mg_el, "name"))
        actors_el = mg_el.find(q("Actors"))
        if actors_el is not None:
            for ref_el in actors_el.findall(q("EntityRef")):
                mg.actors.append(_str(ref_el, "entityRef"))
        for man_el in mg_el.findall(q("Maneuver")):
            mg.maneuvers.append(self._parse_maneuver(man_el, q))
        return mg

    def _parse_maneuver(self, man_el: etree._Element, q: Any) -> Maneuver:
        maneuver = Maneuver(name=_str(man_el, "name"))
        for ev_el in man_el.findall(q("Event")):
            maneuver.events.append(self._parse_event(ev_el, q))
        return maneuver

    def _parse_event(self, ev_el: etree._Element, q: Any) -> Event:
        event = Event(
            name=_str(ev_el, "name"),
            priority=_str(ev_el, "priority", "overwrite"),
        )
        for action_el in ev_el.findall(q("Action")):
            for pa_el in action_el.findall(q("PrivateAction")):
                action = self._parse_private_action(pa_el, q)
                if action is not None:
                    event.actions.append(action)
            for ga_el in action_el.findall(q("GlobalAction")):
                # GlobalActions like EntityAction can be added here in future
                _ = ga_el
        start_el = ev_el.find(q("StartTrigger"))
        if start_el is not None:
            event.start_conditions = self._parse_trigger(start_el, q)
            event.has_start_trigger = True
        return event

    # ------------------------------------------------------------------

    def _parse_private_action(
        self, pa_el: etree._Element, q: Any
    ) -> SpeedAction | LaneChangeAction | TeleportAction | FollowTrajectoryAction | None:
        # TeleportAction
        tel_el = pa_el.find(q("TeleportAction"))
        if tel_el is not None:
            pos_el = tel_el.find(q("Position"))
            if pos_el is not None:
                wp_el = pos_el.find(q("WorldPosition"))
                if wp_el is not None:
                    return TeleportAction(position=self._parse_world_position(wp_el))

        # LongitudinalAction → SpeedAction
        long_el = pa_el.find(q("LongitudinalAction"))
        if long_el is not None:
            speed_el = long_el.find(q("SpeedAction"))
            if speed_el is not None:
                return self._parse_speed_action(speed_el, q)

        # LateralAction → LaneChangeAction
        lat_el = pa_el.find(q("LateralAction"))
        if lat_el is not None:
            lc_el = lat_el.find(q("LaneChangeAction"))
            if lc_el is not None:
                return self._parse_lane_change_action(lc_el, q)

        # RoutingAction → FollowTrajectoryAction
        routing_el = pa_el.find(q("RoutingAction"))
        if routing_el is not None:
            fta_el = routing_el.find(q("FollowTrajectoryAction"))
            if fta_el is not None:
                return self._parse_follow_trajectory_action(fta_el, q)

        return None

    def _parse_follow_trajectory_action(
        self, fta_el: etree._Element, q: Any
    ) -> FollowTrajectoryAction:
        vertices: list[TrajectoryVertex] = []

        # The trajectory may be inline (under TrajectoryRef/Trajectory) or
        # directly under FollowTrajectoryAction as a child Trajectory element.
        traj_el = None
        traj_ref_el = fta_el.find(q("TrajectoryRef"))
        if traj_ref_el is not None:
            traj_el = traj_ref_el.find(q("Trajectory"))
        if traj_el is None:
            traj_el = fta_el.find(q("Trajectory"))

        if traj_el is not None:
            shape_el = traj_el.find(q("Shape"))
            if shape_el is not None:
                polyline_el = shape_el.find(q("Polyline"))
                if polyline_el is not None:
                    for vertex_el in polyline_el.findall(q("Vertex")):
                        time = _float(vertex_el, "time")
                        pos_el = vertex_el.find(q("Position"))
                        if pos_el is not None:
                            wp_el = pos_el.find(q("WorldPosition"))
                            if wp_el is not None:
                                vertices.append(
                                    TrajectoryVertex(
                                        time=time,
                                        position=self._parse_world_position(wp_el),
                                    )
                                )

        time_domain = "relative"
        time_ref_el = fta_el.find(q("TimeReference"))
        if time_ref_el is not None:
            timing_el = time_ref_el.find(q("Timing"))
            if timing_el is not None:
                domain = _str(timing_el, "domainAbsoluteRelative", "relative").lower()
                if domain in {"absolute", "relative"}:
                    time_domain = domain

        # Sort vertices by time to be safe
        vertices.sort(key=lambda v: v.time)
        return FollowTrajectoryAction(vertices=vertices, time_domain=time_domain)

    def _parse_world_position(self, wp_el: etree._Element) -> WorldPosition:
        return WorldPosition(
            x=_float(wp_el, "x"),
            y=_float(wp_el, "y"),
            z=_float(wp_el, "z"),
            h=_float(wp_el, "h"),
            p=_float(wp_el, "p"),
            r=_float(wp_el, "r"),
        )

    def _parse_speed_action(self, speed_el: etree._Element, q: Any) -> SpeedAction:
        dyn_el = speed_el.find(q("SpeedActionDynamics"))
        shape = "step"
        dyn_val = 0.0
        if dyn_el is not None:
            shape = _str(dyn_el, "dynamicsShape", "step")
            dyn_val = _float(dyn_el, "value")

        target = 0.0
        tgt_el = speed_el.find(q("SpeedActionTarget"))
        if tgt_el is not None:
            abs_el = tgt_el.find(q("AbsoluteTargetSpeed"))
            if abs_el is not None:
                target = _float(abs_el, "value")
        return SpeedAction(
            target_speed=target,
            dynamics_shape=shape,
            dynamics_value=dyn_val,
        )

    def _parse_lane_change_action(self, lc_el: etree._Element, q: Any) -> LaneChangeAction:
        dyn_el = lc_el.find(q("LaneChangeActionDynamics"))
        shape = "sinusoidal"
        dyn_val = 3.0
        if dyn_el is not None:
            shape = _str(dyn_el, "dynamicsShape", "sinusoidal")
            dyn_val = _float(dyn_el, "value", 3.0)

        offset = 0.0
        tgt_el = lc_el.find(q("LaneChangeTarget"))
        if tgt_el is not None:
            abs_el = tgt_el.find(q("AbsoluteTargetLane"))
            if abs_el is not None:
                # Treat lane ID delta as lateral offset × 3.5 m lane width
                offset = _float(abs_el, "value") * 3.5
        return LaneChangeAction(
            target_lane_offset=offset,
            dynamics_shape=shape,
            dynamics_value=dyn_val,
        )

    # ------------------------------------------------------------------

    def _parse_trigger(self, trigger_el: etree._Element, q: Any) -> list[Condition]:
        conditions: list[Condition] = []
        for cg_el in trigger_el.findall(q("ConditionGroup")):
            for cond_el in cg_el.findall(q("Condition")):
                cond = self._parse_condition(cond_el, q)
                if cond is not None:
                    conditions.append(cond)
        return conditions

    def _parse_condition(self, cond_el: etree._Element, q: Any) -> Condition | None:
        name = _str(cond_el, "name")
        delay = _float(cond_el, "delay")
        params: dict[str, Any] = {"edge": _str(cond_el, "conditionEdge", "none")}

        # ByValueCondition
        bv_el = cond_el.find(q("ByValueCondition"))
        if bv_el is not None:
            sim_el = bv_el.find(q("SimulationTimeCondition"))
            if sim_el is not None:
                params["type"] = "simulation_time"
                params["value"] = _float(sim_el, "value")
                params["rule"] = _str(sim_el, "rule", "greaterThan")
                return Condition(name=name, delay=delay, params=params)

            ses_el = bv_el.find(q("StoryboardElementStateCondition"))
            if ses_el is not None:
                params["type"] = "storyboard_element_state"
                params["storyboard_element_ref"] = _str(ses_el, "storyboardElementRef")
                params["storyboard_element_type"] = _str(ses_el, "storyboardElementType").lower()
                params["state"] = _str(ses_el, "state")
                return Condition(name=name, delay=delay, params=params)

            param_el = bv_el.find(q("ParameterCondition"))
            if param_el is not None:
                params["type"] = "parameter"
                params["parameter_ref"] = _str(param_el, "parameterRef")
                params["value"] = _str(param_el, "value")
                params["rule"] = _str(param_el, "rule", "equalTo")
                return Condition(name=name, delay=delay, params=params)

        # ByEntityCondition
        be_el = cond_el.find(q("ByEntityCondition"))
        if be_el is not None:
            trig_el = be_el.find(q("TriggeringEntities"))
            if trig_el is not None:
                params["triggering_entities"] = [
                    _str(r, "entityRef") for r in trig_el.findall(q("EntityRef"))
                ]
            ec_el = be_el.find(q("EntityCondition"))
            if ec_el is not None:
                dist_el = ec_el.find(q("DistanceCondition"))
                if dist_el is not None:
                    params["type"] = "distance"
                    params["value"] = _float(dist_el, "value")
                    params["rule"] = _str(dist_el, "rule", "lessThan")
                    params["entity_ref"] = _str(dist_el, "entityRef")
                    # Optional: coordinate system / distance type for longitudinal mode
                    coord_sys = _str(dist_el, "coordinateSystem", "entity")
                    rel_dist_type = _str(dist_el, "relativeDistanceType", "euclidianDistance")
                    params["coordinate_system"] = coord_sys
                    params["relative_distance_type"] = rel_dist_type
                    pos_el = dist_el.find(q("Position"))
                    if pos_el is not None:
                        wp_el = pos_el.find(q("WorldPosition"))
                        if wp_el is not None:
                            params["target_position"] = (
                                _float(wp_el, "x"),
                                _float(wp_el, "y"),
                                _float(wp_el, "z"),
                            )
                    return Condition(name=name, delay=delay, params=params)

                ttc_el = ec_el.find(q("TimeToCollisionCondition"))
                if ttc_el is not None:
                    params["type"] = "ttc"
                    params["value"] = _float(ttc_el, "value")
                    params["rule"] = _str(ttc_el, "rule", "lessThan")
                    params["coordinate_system"] = _str(ttc_el, "coordinateSystem", "entity")
                    params["relative_distance_type"] = _str(
                        ttc_el, "relativeDistanceType", "euclidianDistance"
                    )

                    target_el = ttc_el.find(q("TimeToCollisionConditionTarget"))
                    if target_el is not None:
                        target_ref = target_el.get("entityRef")
                        if target_ref is not None:
                            params["entity_ref"] = target_ref
                        pos_el = target_el.find(q("Position"))
                        if pos_el is not None:
                            wp_el = pos_el.find(q("WorldPosition"))
                            if wp_el is not None:
                                params["target_position"] = (
                                    _float(wp_el, "x"),
                                    _float(wp_el, "y"),
                                    _float(wp_el, "z"),
                                )
                    return Condition(name=name, delay=delay, params=params)

                speed_el = ec_el.find(q("SpeedCondition"))
                if speed_el is not None:
                    params["type"] = "speed"
                    params["value"] = _float(speed_el, "value")
                    params["rule"] = _str(speed_el, "rule", "greaterThan")
                    return Condition(name=name, delay=delay, params=params)

                dist_travel_el = ec_el.find(q("TraveledDistanceCondition"))
                if dist_travel_el is not None:
                    params["type"] = "traveled_distance"
                    params["value"] = _float(dist_travel_el, "value")
                    return Condition(name=name, delay=delay, params=params)

        return None
