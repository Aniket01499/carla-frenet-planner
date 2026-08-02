"""
ego_camera.py

Single entry point. Spawns background traffic, an ego vehicle (autopilot --
driving accuracy is out of scope, see conversation notes), the 5-sensor
suite defined in sensor_config.py via SensorManager, and runs the existing
Frenet-frame trajectory planner alongside it for visualization (shadow-only:
it draws the chosen path, it doesn't drive the car -- that's a separate
follow-up).

Every ~1 second, logs a one-line-per-sensor diagnostic (point/detection
counts and ranges) instead of the old per-tick trajectory print spam.

Run with the CARLA server already running. Press 'q' in any sensor window
to stop. Requires sensor_manager.py, sensor_config.py, and
sensor_visualization.py in the same folder.
"""

from __future__ import annotations

import logging
import math
import random

import carla
import cv2

from global_planner import get_forward_centerline, visualize_path
from frenet_math import get_frenet, get_cartesian
from trajectory_generator import generate_frenet_paths
from cost_function import CostEvaluator
from sensor_manager import SensorManager, summarize_frame
from sensor_config import SENSOR_SPECS
from sensor_visualization import (
    rgb_to_bgr_array,
    semantic_to_bgr_array,
    lidar_to_bev,
    radar_to_bev,
    fused_bev,
)
from calibration import lidar_to_vehicle_frame, radar_to_vehicle_frame

SPEC_BY_NAME = {spec.name: spec for spec in SENSOR_SPECS}

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DIAGNOSTIC_INTERVAL_TICKS = 20  # ~1 second at fixed_delta_seconds=0.05


def spawn_background_traffic(client, world, blueprint_library, spawn_points, count=50):
    """Batch-spawn NPC traffic under Traffic Manager control. Returns actor ids."""
    blueprints_vehicles = blueprint_library.filter("vehicle.*")
    blueprints_vehicles = [x for x in blueprints_vehicles if int(x.get_attribute("number_of_wheels")) == 4]

    random.shuffle(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    tm_port = client.get_trafficmanager(8000).get_port()
    batch = []
    for transform in spawn_points[:count]:
        bp = random.choice(blueprints_vehicles)
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm_port)))

    responses = client.apply_batch_sync(batch, True)
    npc_ids = [r.actor_id for r in responses if not r.error]
    logger.info(f"Successfully spawned {len(npc_ids)} traffic vehicles.")
    return npc_ids


def run_planning_step(world, vehicle, cost_evaluator):
    """One tick of the shadow planner: compute + visualize the best path only."""
    map_x, map_y, map_s = get_forward_centerline(vehicle, world.get_map())
    visualize_path(world, map_x, map_y)

    if len(map_s) < 2:
        return

    transform = vehicle.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x ** 2 + vel.y ** 2)

    c_s, c_d = get_frenet(loc.x, loc.y, yaw, map_x, map_y, map_s)
    bundles = generate_frenet_paths(c_s=c_s, c_s_v=speed, c_s_a=0.0,
                                     c_d=c_d, c_d_v=0.0, c_d_a=0.0)

    best_path, lowest_cost = None, float("inf")
    obstacles = []  # ground-truth obstacle wiring is a follow-up step
    for path in bundles:
        cost = cost_evaluator.calculate_total_cost(path, obstacles, target_speed=20.0)
        if cost < lowest_cost:
            lowest_cost, best_path = cost, path

    if best_path:
        for i in range(0, len(best_path["s_points"]), 3):
            s, d = best_path["s_points"][i], best_path["d_points"][i]
            p_x, p_y = get_cartesian(s, d, map_x, map_y, map_s)
            world.debug.draw_point(
                carla.Location(x=p_x, y=p_y, z=loc.z + 0.5),
                size=0.1, color=carla.Color(255, 0, 0), life_time=0.1,
            )


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    # Ego vehicle
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[10] if len(spawn_points) > 10 else spawn_points[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, tm.get_port())
    logger.info("Ego vehicle spawned!")

    # Background traffic
    remaining_spawn_points = [p for p in spawn_points if p != spawn_point]
    npc_ids = spawn_background_traffic(client, world, blueprint_library, remaining_spawn_points)

    # Sensors
    manager = SensorManager(world, vehicle, SENSOR_SPECS, output_root="runs")
    manager.spawn_all()

    cost_evaluator = CostEvaluator()

    logger.info("Streaming in SYNCHRONOUS MODE. Press 'q' in a sensor window to stop.")
    tick_count = 0
    try:
        while True:
            world.tick()
            frame_id = world.get_snapshot().frame
            readings = manager.get_synced_frame(frame_id, timeout=2.0)

            cv2.imshow("Left Blind Spot", rgb_to_bgr_array(readings["left_blind_spot_rgb"]))
            cv2.imshow("Right Blind Spot", rgb_to_bgr_array(readings["right_blind_spot_rgb"]))
            cv2.imshow("Center Semantic", semantic_to_bgr_array(readings["center_semantic"]))
            cv2.imshow("Lidar BEV", lidar_to_bev(readings["roof_lidar"]))
            cv2.imshow("Radar BEV", radar_to_bev(readings["front_radar"]))

            lidar_vehicle = lidar_to_vehicle_frame(
                readings["roof_lidar"], SPEC_BY_NAME["roof_lidar"].transform)
            radar_vehicle = radar_to_vehicle_frame(
                readings["front_radar"], SPEC_BY_NAME["front_radar"].transform)
            cv2.imshow("Fused BEV", fused_bev(lidar_vehicle, radar_vehicle))

            run_planning_step(world, vehicle, cost_evaluator)

            tick_count += 1
            if tick_count % DIAGNOSTIC_INTERVAL_TICKS == 0:
                for name, line in summarize_frame(readings).items():
                    logger.info(f"[{name}] {line}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user...")
    finally:
        logger.info("Cleaning up...")
        manager.destroy_all()
        vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        cv2.destroyAllWindows()

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
