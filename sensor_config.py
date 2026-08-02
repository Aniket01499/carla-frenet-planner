"""
sensor_config.py

Single source of truth for the ego vehicle's sensor suite. Every script
that needs the sensor layout imports SENSOR_SPECS from here -- this is
the fix for the exact duplication bug (same list copy-pasted across
sensor_manager.py and multi_sensor_viewer.py) that caused silent drift
in the old CARISSMA scripts. multi_sensor_viewer.py is now superseded
by ego_camera.py and can be deleted.
"""

from __future__ import annotations

import carla

from sensor_manager import SensorSpec

SENSOR_SPECS = [
    SensorSpec(
        name="left_blind_spot_rgb",
        blueprint_id="sensor.camera.rgb",
        transform=carla.Transform(carla.Location(x=0.7, y=-0.9, z=1.1),
                                   carla.Rotation(yaw=-150)),
        attributes={"image_size_x": 480, "image_size_y": 270, "fov": 100},
    ),
    SensorSpec(
        name="right_blind_spot_rgb",
        blueprint_id="sensor.camera.rgb",
        transform=carla.Transform(carla.Location(x=0.7, y=0.9, z=1.1),
                                   carla.Rotation(yaw=150)),
        attributes={"image_size_x": 480, "image_size_y": 270, "fov": 100},
    ),
    SensorSpec(
        name="center_semantic",
        blueprint_id="sensor.camera.semantic_segmentation",
        transform=carla.Transform(carla.Location(x=1.5, z=1.4)),
        attributes={"image_size_x": 640, "image_size_y": 360, "fov": 90},
    ),
    SensorSpec(
        name="roof_lidar",
        blueprint_id="sensor.lidar.ray_cast",
        transform=carla.Transform(carla.Location(x=0.0, z=2.4)),
        attributes={
            "channels": 32, "range": 50, "points_per_second": 100000,
            "rotation_frequency": 20,
            # CARLA's defaults randomly discard ~45% of points
            # (dropoff_general_rate=0.45). Zeroed out here so the point
            # count we log is the real number the raycast produced, not
            # thinned by dropoff -- rules dropoff in/out as a cause of
            # the sparse cloud before touching anything else.
            "dropoff_general_rate": 0.0,
            "dropoff_intensity_limit": 1.0,
            "dropoff_zero_intensity": 0.0,
        },
    ),
    SensorSpec(
        name="front_radar",
        blueprint_id="sensor.other.radar",
        transform=carla.Transform(carla.Location(x=2.0, z=1.0)),
        attributes={"horizontal_fov": 30, "vertical_fov": 10, "range": 70},
    ),
]
