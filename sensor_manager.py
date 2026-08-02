"""
sensor_manager.py

Config-driven multi-sensor management for CARLA scenario capture.
Owns a set of sensors attached to one vehicle, keeps them synchronized
to the simulation tick, and writes structured output to disk.

Compatible with Python 3.8 (matches CARLA 0.9.13's required interpreter).
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import carla
import numpy as np


@dataclass
class SensorSpec:
    """Declarative definition of a single sensor to attach to the ego vehicle."""
    name: str                                  # unique identifier, e.g. "front_rgb"
    blueprint_id: str                          # e.g. "sensor.camera.rgb"
    transform: carla.Transform                 # mount pose relative to the vehicle
    attributes: Dict[str, Any] = field(default_factory=dict)  # e.g. {"image_size_x": 1280}


class SensorManager:
    """
    Usage:
        specs = [SensorSpec(...), SensorSpec(...)]
        manager = SensorManager(world, vehicle, specs, output_root="runs")
        manager.spawn_all()
        try:
            for _ in range(n_frames):
                world.tick()
                frame_id = world.get_snapshot().frame
                readings = manager.get_synced_frame(frame_id, timeout=2.0)
                manager.save_frame(readings, frame_id)
        finally:
            manager.destroy_all()
    """

    def __init__(self, world: carla.World, vehicle: carla.Actor,
                 specs: List[SensorSpec], output_root: str = "runs"):
        self.world = world
        self.vehicle = vehicle
        self.specs = specs
        self.actors: Dict[str, carla.Actor] = {}
        self.queues: Dict[str, "queue.Queue[Any]"] = {}

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(output_root) / timestamp
        for spec in specs:
            (self.run_dir / spec.name).mkdir(parents=True, exist_ok=True)

    def spawn_all(self) -> None:
        """Spawn every configured sensor and attach a listener queue."""
        bp_library = self.world.get_blueprint_library()
        for spec in self.specs:
            bp = bp_library.find(spec.blueprint_id)
            for attr_name, attr_value in spec.attributes.items():
                bp.set_attribute(attr_name, str(attr_value))

            actor = self.world.spawn_actor(bp, spec.transform, attach_to=self.vehicle)
            q: "queue.Queue[Any]" = queue.Queue()
            actor.listen(q.put)

            self.actors[spec.name] = actor
            self.queues[spec.name] = q

    def get_synced_frame(self, frame_id: int, timeout: float = 2.0) -> Dict[str, Any]:
        """
        Pull exactly one reading per sensor matching the given simulation
        frame_id. Call this once, right after world.tick(), before doing
        anything else that might drain the queues.

        Discards stale readings from earlier ticks; raises queue.Empty if a
        sensor doesn't produce a matching frame within `timeout` seconds
        (this usually means a sensor's own tick rate is slower than the
        simulation tick -- worth checking sensor_tick/rotation_frequency
        settings if you hit this).
        """
        readings: Dict[str, Any] = {}
        for name, q in self.queues.items():
            while True:
                data = q.get(timeout=timeout)
                if data.frame == frame_id:
                    readings[name] = data
                    break
                # else: stale reading from a previous tick, discard and retry
        return readings

    def save_frame(self, readings: Dict[str, Any], frame_id: int) -> None:
        """Write one synced frame to disk, dispatching by sensor data type."""
        for name, data in readings.items():
            out_dir = self.run_dir / name
            if isinstance(data, carla.Image):
                data.save_to_disk(str(out_dir / f"{frame_id:06d}.png"))
            elif isinstance(data, carla.LidarMeasurement):
                points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                np.save(out_dir / f"{frame_id:06d}.npy", points)  # x, y, z, intensity
            elif isinstance(data, carla.RadarMeasurement):
                detections = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                np.save(out_dir / f"{frame_id:06d}.npy", detections)
            else:
                raise TypeError(f"Unhandled sensor data type for '{name}': {type(data)}")

    def destroy_all(self) -> None:
        """Stop and destroy every spawned sensor. Always call from a finally block."""
        for actor in self.actors.values():
            actor.stop()
            actor.destroy()
        self.actors.clear()
        self.queues.clear()


def summarize_frame(readings: Dict[str, Any]) -> Dict[str, str]:
    """
    One-line-per-sensor diagnostic summary of a synced frame: point/detection
    counts and value ranges, not a full data dump. Meant to answer "is this
    sensor actually producing data, and does it look sane" without staring
    at a mostly-black BEV window and guessing.
    """
    summary: Dict[str, str] = {}
    for name, data in readings.items():
        if isinstance(data, carla.Image):
            summary[name] = f"image {data.width}x{data.height}, frame {data.frame}"

        elif isinstance(data, carla.LidarMeasurement):
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            if points.size:
                summary[name] = (
                    f"{len(points)} points | "
                    f"x[{points[:,0].min():.1f},{points[:,0].max():.1f}] "
                    f"y[{points[:,1].min():.1f},{points[:,1].max():.1f}] "
                    f"z[{points[:,2].min():.1f},{points[:,2].max():.1f}]"
                )
            else:
                summary[name] = "0 points (!)"

        elif isinstance(data, carla.RadarMeasurement):
            detections = list(data)
            if detections:
                velocities = [d.velocity for d in detections]
                azimuths_deg = [math.degrees(d.azimuth) for d in detections]
                depths = [d.depth for d in detections]
                summary[name] = (
                    f"{len(detections)} detections | "
                    f"velocity[{min(velocities):.1f},{max(velocities):.1f}]m/s "
                    f"azimuth[{min(azimuths_deg):.1f},{max(azimuths_deg):.1f}]deg "
                    f"depth[{min(depths):.1f},{max(depths):.1f}]m"
                )
            else:
                summary[name] = "0 detections (!)"

        else:
            summary[name] = f"unhandled type {type(data)}"

    return summary
