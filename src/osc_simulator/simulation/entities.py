"""Runtime entity state for the simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from osc_simulator.parser.openscenario import EntityDef, TrajectoryVertex, WorldPosition


@dataclass
class DynamicsProfile:
    """Active speed-change interpolation."""
    initial_speed: float = 0.0
    target_speed: float = 0.0
    shape: str = "step"        # "step" | "linear" | "sinusoidal"
    duration: float = 0.0      # seconds (0 → immediate)
    elapsed: float = 0.0

    def is_complete(self) -> bool:
        return self.duration <= 0.0 or self.elapsed >= self.duration

    def current_speed(self) -> float:
        if self.is_complete() or self.duration <= 0.0:
            return self.target_speed
        t = self.elapsed / self.duration
        if self.shape == "linear":
            return self.initial_speed + t * (self.target_speed - self.initial_speed)
        if self.shape == "sinusoidal":
            s = 0.5 * (1.0 - math.cos(math.pi * t))
            return self.initial_speed + s * (self.target_speed - self.initial_speed)
        return self.target_speed  # step


@dataclass
class LateralProfile:
    """Active lateral-offset interpolation."""
    initial_offset: float = 0.0
    target_offset: float = 0.0
    shape: str = "sinusoidal"
    duration: float = 3.0
    elapsed: float = 0.0

    def is_complete(self) -> bool:
        return self.elapsed >= self.duration

    def current_offset(self) -> float:
        if self.is_complete():
            return self.target_offset
        t = self.elapsed / self.duration
        s = 0.5 * (1.0 - math.cos(math.pi * t)) if self.shape == "sinusoidal" else t
        return self.initial_offset + s * (self.target_offset - self.initial_offset)


@dataclass
class EntityRuntimeState:
    definition: EntityDef

    # Current kinematic state
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0   # radians, 0 = East (+x), positive counter-clockwise
    speed: float = 0.0     # m/s longitudinal
    lateral_offset: float = 0.0  # metres from initial lane centre
    odometer: float = 0.0  # total distance travelled (metres)

    # Active interpolations
    dynamics: DynamicsProfile | None = None
    lateral: LateralProfile | None = None

    # Active trajectory (FollowTrajectoryAction)
    trajectory: list[TrajectoryVertex] | None = None
    # current simulation clock used for trajectory interpolation
    _trajectory_time: float = field(default=0.0, repr=False)

    @classmethod
    def from_definition(cls, defn: EntityDef) -> EntityRuntimeState:
        s = defn.initial_state
        p = s.position
        return cls(
            definition=defn,
            x=p.x,
            y=p.y,
            z=p.z,
            heading=p.h,
            speed=s.speed,
        )

    def apply_speed_action(
        self, target: float, shape: str = "step", duration: float = 0.0
    ) -> None:
        self.dynamics = DynamicsProfile(
            initial_speed=self.speed,
            target_speed=target,
            shape=shape,
            duration=duration,
        )
        if shape == "step":
            self.speed = target
            self.dynamics = None

    def apply_lateral_action(
        self, target_offset: float, shape: str = "sinusoidal", duration: float = 3.0
    ) -> None:
        self.lateral = LateralProfile(
            initial_offset=self.lateral_offset,
            target_offset=target_offset,
            shape=shape,
            duration=duration,
        )

    def apply_teleport(self, pos: WorldPosition) -> None:
        self.x = pos.x
        self.y = pos.y
        self.z = pos.z
        self.heading = pos.h

    def apply_trajectory(self, vertices: list[TrajectoryVertex]) -> None:
        """Activate a polyline trajectory (absolute time references)."""
        self.trajectory = vertices
        # Seed position/heading from the first vertex immediately
        if vertices:
            first = vertices[0]
            self.x = first.position.x
            self.y = first.position.y
            self.z = first.position.z
            self.heading = first.position.h
            self._trajectory_time = first.time

    def _interpolate_trajectory(self, t: float) -> None:
        """Update position/heading/speed by linearly interpolating the polyline at time *t*."""
        verts = self.trajectory
        if not verts:
            return

        # Before first vertex — clamp to first
        if t <= verts[0].time:
            p = verts[0].position
            self.x = p.x
            self.y = p.y
            self.z = p.z
            self.heading = p.h
            self.speed = 0.0
            return

        # After last vertex — clamp to last
        if t >= verts[-1].time:
            p = verts[-1].position
            self.x = p.x
            self.y = p.y
            self.z = p.z
            self.heading = p.h
            self.speed = 0.0
            return

        # Find the bracketing vertices
        for i in range(len(verts) - 1):
            t0, t1 = verts[i].time, verts[i + 1].time
            if t0 <= t <= t1:
                dt_seg = t1 - t0
                alpha = (t - t0) / dt_seg if dt_seg > 0.0 else 0.0
                p0, p1 = verts[i].position, verts[i + 1].position

                self.x = p0.x + alpha * (p1.x - p0.x)
                self.y = p0.y + alpha * (p1.y - p0.y)
                self.z = p0.z + alpha * (p1.z - p0.z)

                # Heading: derive from segment direction, fall back to stored h
                dx = p1.x - p0.x
                dy = p1.y - p0.y
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-9:
                    self.heading = math.atan2(dy, dx)
                else:
                    self.heading = p0.h + alpha * (p1.h - p0.h)

                # Speed: Euclidean distance over time
                seg_dist = math.sqrt(dx * dx + dy * dy + (p1.z - p0.z) ** 2)
                self.speed = seg_dist / dt_seg if dt_seg > 0.0 else 0.0
                return

    def step(self, dt: float) -> None:
        prev_x, prev_y, prev_z = self.x, self.y, self.z

        if self.trajectory is not None:
            # Advance the trajectory clock and interpolate
            self._trajectory_time += dt
            self._interpolate_trajectory(self._trajectory_time)
        else:
            # Advance dynamics interpolation
            if self.dynamics is not None:
                self.dynamics.elapsed += dt
                self.speed = self.dynamics.current_speed()
                if self.dynamics.is_complete():
                    self.speed = self.dynamics.target_speed
                    self.dynamics = None

            # Advance lateral interpolation
            if self.lateral is not None:
                self.lateral.elapsed += dt
                self.lateral_offset = self.lateral.current_offset()
                if self.lateral.is_complete():
                    self.lateral_offset = self.lateral.target_offset
                    self.lateral = None

            # Integrate position (simple Euler forward)
            self.x += self.speed * math.cos(self.heading) * dt
            self.y += self.speed * math.sin(self.heading) * dt
            # Apply lateral offset perpendicular to heading
            lateral_heading = self.heading + math.pi / 2.0
            self.x += self.lateral_offset * math.cos(lateral_heading) * dt
            self.y += self.lateral_offset * math.sin(lateral_heading) * dt

        # Accumulate odometer
        self.odometer += math.hypot(self.x - prev_x, self.y - prev_y, self.z - prev_z)
