"""
calibration.py

Extrinsic calibration: converts raw per-sensor readings (given in each
sensor's own local frame) into the vehicle's frame, using the sensor's
mount transform (SensorSpec.transform). That transform already IS the
extrinsic calibration -- position and orientation relative to the vehicle
-- it just hasn't been applied to the actual point data until now.

Without this step, lidar and radar each report coordinates relative to
their own mount point, facing their own mount orientation -- two points
that are physically the same spot in the world can have completely
different (x, y, z) in each sensor's raw output. This module is what
makes them comparable.
"""

from __future__ import annotations

import math

import carla
import numpy as np


def _transform_matrix(sensor_transform: carla.Transform) -> np.ndarray:
    """4x4 homogeneous transform matrix: sensor-local -> vehicle frame."""
    return np.array(sensor_transform.get_matrix())


def lidar_to_vehicle_frame(lidar_measurement: carla.LidarMeasurement,
                            sensor_transform: carla.Transform) -> np.ndarray:
    """Returns (N, 3) array of lidar points in the vehicle's frame."""
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    xyz_local = points[:, :3]
    if xyz_local.shape[0] == 0:
        return xyz_local

    matrix = _transform_matrix(sensor_transform)
    xyz_homogeneous = np.hstack([xyz_local, np.ones((xyz_local.shape[0], 1), dtype=np.float32)])
    xyz_vehicle = (matrix @ xyz_homogeneous.T).T[:, :3]
    return xyz_vehicle


def radar_to_vehicle_frame(radar_measurement: carla.RadarMeasurement,
                            sensor_transform: carla.Transform) -> np.ndarray:
    """
    Returns (N, 4) array: x, y, z in the vehicle's frame, velocity unchanged.
    Velocity is a scalar range-rate (how fast the detection is closing or
    opening relative to the radar) -- a rotation/translation of the frame
    doesn't change that number, so it's carried straight through.
    """
    rows = []
    for detection in radar_measurement:
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        rows.append((x, y, z, detection.velocity))

    if not rows:
        return np.zeros((0, 4), dtype=np.float32)

    local = np.array(rows, dtype=np.float32)
    matrix = _transform_matrix(sensor_transform)
    xyz_homogeneous = np.hstack([local[:, :3], np.ones((local.shape[0], 1), dtype=np.float32)])
    xyz_vehicle = (matrix @ xyz_homogeneous.T).T[:, :3]
    return np.hstack([xyz_vehicle, local[:, 3:4]])
