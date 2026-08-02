"""
multi_sensor_viewer.py

Spawns an ego vehicle (autopilot -- planning/control accuracy is out of
scope for this step) with front RGB, front semantic segmentation, roof
lidar, and front radar sensors, and displays each in its own OpenCV window.

Run with the CARLA server already running. Press 'q' in any window to stop.
"""

from __future__ import annotations

import cv2
import carla

from sensor_manager import SensorSpec, SensorManager
from sensor_visualization import (
    rgb_to_bgr_array,
    semantic_to_bgr_array,
    lidar_to_bev,
    radar_to_bev,
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
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, tm.get_port())

    manager = SensorManager(world, vehicle, SENSOR_SPECS, output_root="runs")
    manager.spawn_all()

    print("Streaming. Press 'q' in any window to stop.")
    try:
        while True:
            world.tick()
            frame_id = world.get_snapshot().frame
            readings = manager.get_synced_frame(frame_id, timeout=2.0)

            cv2.imshow("Front RGB", rgb_to_bgr_array(readings["front_rgb"]))
            cv2.imshow("Front Semantic", semantic_to_bgr_array(readings["front_semantic"]))
            cv2.imshow("Lidar BEV", lidar_to_bev(readings["roof_lidar"]))
            cv2.imshow("Radar BEV", radar_to_bev(readings["front_radar"]))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
        manager.destroy_all()
        vehicle.destroy()
        cv2.destroyAllWindows()

        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
