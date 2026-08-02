"""
ego_camera.py

Single entry point. Traffic Manager autopilot handles navigation (route
choice, when to slow/turn/stop) exactly as it would for any CARLA vehicle
-- that's not the point of this project. What IS the point: instead of
letting CARLA's PhysX execute the throttle/steer/brake that autopilot
computes, we intercept it and run it through vehicle_dynamics.py (the
thesis's Fiala-tire bicycle model) each tick, then push the result back
into CARLA via set_transform(). CARLA is now used purely for navigation
decisions, rendering, and sensors -- not vehicle physics.

The existing Frenet planner still runs and draws its chosen path as
debug dots each tick (uses ground-truth NPC positions, not a perception
model) -- it's a visualization/portfolio piece now, not connected to
vehicle control.

Run with the CARLA server already running. Press 'q' in any sensor window
to stop. Requires sensor_manager.py, sensor_config.py,
sensor_visualization.py, calibration.py, and vehicle_dynamics.py in the
same folder.
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
from vehicle_dynamics import step as dynamics_step, VehicleParams, BicycleState

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SPEC_BY_NAME = {spec.name: spec for spec in SENSOR_SPECS}
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


def get_ground_truth_obstacles(world, ego_vehicle, map_x, map_y, map_s):
    """Ground-truth NPC positions (not a perception model), projected to Frenet."""
    obstacles = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_vehicle.id:
            continue
        loc = actor.get_location()
        yaw = math.radians(actor.get_transform().rotation.yaw)
        try:
            s, d = get_frenet(loc.x, loc.y, yaw, map_x, map_y, map_s)
        except (IndexError, ZeroDivisionError):
            continue
        obstacles.append({"s": s, "d": d})
    return obstacles


def visualize_shadow_plan(world, vehicle, cost_evaluator):
    """
    Portfolio piece, not connected to vehicle control: runs the Frenet
    planner against ground-truth obstacles and draws the chosen path.
    Doesn't drive the car -- Traffic Manager + vehicle_dynamics.py do that.
    """
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
    obstacles = get_ground_truth_obstacles(world, vehicle, map_x, map_y, map_s)

    best_path, lowest_cost = None, float("inf")
    for path in bundles:
        cost = cost_evaluator.calculate_total_cost(path, obstacles, target_speed=15.0)
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


def autopilot_control_to_dynamics_input(vehicle, max_steer_rad):
    """
    Reads whatever throttle/steer/brake Traffic Manager just computed for
    this vehicle -- apply_control() sets that state regardless of whether
    physics is enabled to act on it -- and converts it into the throttle
    (single signed value) / steering (radians) convention
    vehicle_dynamics.step() expects.
    """
    control = vehicle.get_control()
    throttle = -control.brake if control.brake > 0.0 else control.throttle
    steering = control.steer * max_steer_rad
    return throttle, steering, control


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

    # Ego vehicle: autopilot handles navigation, our model handles physics.
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[10] if len(spawn_points) > 10 else spawn_points[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Grab the vehicle's real max steering angle before disabling physics --
    # control.steer is normalized [-1, 1], not radians, and this converts it.
    physics_control = vehicle.get_physics_control()
    front_wheels = physics_control.wheels[:2]
    max_steer_rad = math.radians(sum(w.max_steer_angle for w in front_wheels) / len(front_wheels))
    logger.info(f"Vehicle max steer angle: {math.degrees(max_steer_rad):.1f} deg")

    vehicle.set_autopilot(True, tm.get_port())
    vehicle.set_simulate_physics(False)
    logger.info("Autopilot ON for navigation. Physics OFF -- vehicle_dynamics.py executes it.")

    spawn_z = spawn_point.location.z
    dyn_state = BicycleState(
        X=spawn_point.location.x,
        Y=spawn_point.location.y,
        psi=math.radians(spawn_point.rotation.yaw),
        v=0.0, r=0.0, beta=0.0,
    )
    dyn_params = VehicleParams()

    remaining_spawn_points = [p for p in spawn_points if p != spawn_point]
    npc_ids = spawn_background_traffic(client, world, blueprint_library, remaining_spawn_points)

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

            visualize_shadow_plan(world, vehicle, cost_evaluator)

            throttle, steering, raw_control = autopilot_control_to_dynamics_input(vehicle, max_steer_rad)
            dyn_state = dynamics_step(dyn_state, throttle, steering, settings.fixed_delta_seconds, dyn_params)

            vehicle.set_transform(carla.Transform(
                carla.Location(x=dyn_state.X, y=dyn_state.Y, z=spawn_z),
                carla.Rotation(yaw=math.degrees(dyn_state.psi)),
            ))
            vehicle.set_target_velocity(carla.Vector3D(
                dyn_state.v * math.cos(dyn_state.psi + dyn_state.beta),
                dyn_state.v * math.sin(dyn_state.psi + dyn_state.beta),
                0.0,
            ))

            tick_count += 1
            if tick_count % DIAGNOSTIC_INTERVAL_TICKS == 0:
                for name, line in summarize_frame(readings).items():
                    logger.info(f"[{name}] {line}")
                logger.info(f"[autopilot_control] throttle={raw_control.throttle:.2f} "
                            f"steer={raw_control.steer:.2f} brake={raw_control.brake:.2f}")
                logger.info(f"[ego_dynamics] v={dyn_state.v:.2f}m/s "
                            f"steering={math.degrees(steering):.1f}deg beta={math.degrees(dyn_state.beta):.2f}deg")

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
