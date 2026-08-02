"""
sensor_visualization.py

Pure conversion functions: raw CARLA sensor readings -> displayable numpy
arrays. No CARLA world access, no cv2.imshow (window display) calls here --
cv2's drawing primitives (circle/line/putText) are used to annotate the
array itself, which is still a deterministic, testable array operation,
not a display side effect.

BEV convention: sensor sits near the bottom-center of the canvas, forward
distance maps to "up" (decreasing pixel row), lateral offset maps to
left/right -- the standard way forward-facing automotive sensor data is
displayed.
"""

from __future__ import annotations

import math

import carla
import cv2
import numpy as np

SEMANTIC_COLORS = {
    1: (70, 70, 70),    # Buildings
    4: (0, 0, 255),     # Pedestrians
    7: (128, 64, 128),  # Roads
    10: (255, 0, 0),    # Vehicles
}


def rgb_to_bgr_array(image: carla.Image) -> np.ndarray:
    """carla.Image (RGB camera) -> displayable BGR array for cv2.imshow."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3]


def semantic_to_bgr_array(image: carla.Image) -> np.ndarray:
    """carla.Image (semantic segmentation camera) -> color-coded BGR array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    tags = array[:, :, 2]

    view = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for tag_id, color in SEMANTIC_COLORS.items():
        view[tags == tag_id] = color
    return view


def _draw_bev_reference(canvas: np.ndarray, origin_x: int, origin_y: int,
                         range_m: float, scale: float, fov_deg: float | None = None) -> None:
    """
    Draws faint range rings, an FOV wedge (if given), and an ego-vehicle
    marker onto a BEV canvas in place, so scattered points/detections have
    spatial context instead of floating with no sense of scale or heading.
    """
    ring_color = (40, 40, 40)
    ring_step_m = 20
    for ring_m in range(ring_step_m, int(range_m) + 1, ring_step_m):
        radius_px = int(ring_m * scale)
        cv2.circle(canvas, (origin_x, origin_y), radius_px, ring_color, 1)
        label_y = origin_y - radius_px + 12
        if 0 <= label_y < canvas.shape[0]:
            cv2.putText(canvas, f"{ring_m}m", (origin_x + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ring_color, 1)

    if fov_deg is not None:
        half = math.radians(fov_deg / 2)
        length_px = int(range_m * scale)
        for sign in (-1, 1):
            end_x = origin_x + int(length_px * math.sin(sign * half))
            end_y = origin_y - int(length_px * math.cos(sign * half))
            cv2.line(canvas, (origin_x, origin_y), (end_x, end_y), ring_color, 1)

    cv2.drawMarker(canvas, (origin_x, origin_y), (0, 200, 200),
                    markerType=cv2.MARKER_TRIANGLE_UP, markerSize=10, thickness=2)


def lidar_to_bev(lidar_measurement: carla.LidarMeasurement,
                  size_px: int = 600, range_m: float = 50.0) -> np.ndarray:
    """
    Bird's-eye-view render of a lidar point cloud. Sensor stays centered
    (this is a 360-degree sensor, unlike radar), forward = up.
    """
    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    origin_x, origin_y = size_px // 2, size_px // 2
    scale = size_px / (2 * range_m)

    _draw_bev_reference(canvas, origin_x, origin_y, range_m, scale, fov_deg=None)

    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    if points.size == 0:
        return canvas

    px = (points[:, 1] * scale + origin_x).astype(np.int32)   # lateral -> horizontal
    py = (-points[:, 0] * scale + origin_y).astype(np.int32)  # forward -> up

    in_bounds = (px >= 0) & (px < size_px) & (py >= 0) & (py < size_px)
    px, py = px[in_bounds], py[in_bounds]

    canvas[py, px] = (0, 255, 0)  # flat bright green -- presence, not height
    return canvas


def radar_to_bev(radar_measurement: carla.RadarMeasurement,
                  size_px: int = 600, range_m: float = 70.0, fov_deg: float = 30.0) -> np.ndarray:
    """
    Bird's-eye-view render of radar detections. Sensor sits near the
    bottom-center (forward-only sensor), forward = up.
    Color: red = approaching, blue = receding.
    fov_deg should match the sensor's horizontal_fov attribute, so the
    drawn wedge actually reflects what the radar can see.
    """
    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    origin_x, origin_y = size_px // 2, size_px - 20
    scale = size_px / range_m

    _draw_bev_reference(canvas, origin_x, origin_y, range_m, scale, fov_deg=fov_deg)

    for detection in radar_measurement:
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)  # forward
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)  # lateral

        px = int(origin_x + y * scale)
        py = int(origin_y - x * scale)

        if 0 <= px < size_px and 0 <= py < size_px:
            color = (0, 0, 255) if detection.velocity < 0 else (255, 0, 0)
            cv2.circle(canvas, (px, py), 4, color, -1)

    return canvas


def fused_bev(lidar_points_vehicle: np.ndarray, radar_points_vehicle: np.ndarray,
              size_px: int = 600, range_m: float = 50.0) -> np.ndarray:
    """
    Both inputs must already be in the vehicle's frame (see calibration.py)
    -- that's the actual point of calibration: before it, lidar and radar
    each had their own private coordinate system and had no meaningful way
    to be drawn on the same canvas at all. This is spatial fusion (a shared
    frame), not track-level fusion (associating a specific lidar cluster
    with a specific radar detection over time) -- that's a further step.
    """
    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    origin_x, origin_y = size_px // 2, size_px // 2
    scale = size_px / (2 * range_m)

    _draw_bev_reference(canvas, origin_x, origin_y, range_m, scale, fov_deg=None)

    if lidar_points_vehicle.size:
        px = (lidar_points_vehicle[:, 1] * scale + origin_x).astype(np.int32)
        py = (-lidar_points_vehicle[:, 0] * scale + origin_y).astype(np.int32)
        in_bounds = (px >= 0) & (px < size_px) & (py >= 0) & (py < size_px)
        canvas[py[in_bounds], px[in_bounds]] = (0, 140, 0)  # dimmer green -- radar sits visually on top

    for x, y, z, velocity in radar_points_vehicle:
        px = int(y * scale + origin_x)
        py = int(-x * scale + origin_y)
        if 0 <= px < size_px and 0 <= py < size_px:
            color = (0, 0, 255) if velocity < 0 else (255, 0, 0)
            cv2.circle(canvas, (px, py), 4, color, -1)

    return canvas

