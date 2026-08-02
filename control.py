"""
control.py

Pure Pursuit steering controller (Coulter, Carnegie Mellon, 1992) + a
simple proportional speed controller. Converts a chosen path (already
selected by cost_function.py) into throttle/steering commands for
vehicle_dynamics.step().

No CARLA dependency -- pure geometry and arithmetic, testable in isolation.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def find_lookahead_point(path_x: List[float], path_y: List[float],
                          vehicle_x: float, vehicle_y: float,
                          lookahead_m: float) -> Tuple[float, float]:
    """
    Returns the first path point at least lookahead_m ahead of the vehicle.
    Falls back to the last path point if the whole path is shorter than
    the lookahead distance.
    """
    for x, y in zip(path_x, path_y):
        if math.hypot(x - vehicle_x, y - vehicle_y) >= lookahead_m:
            return x, y
    return path_x[-1], path_y[-1]


def pure_pursuit_steering(vehicle_x: float, vehicle_y: float, vehicle_yaw: float,
                           target_x: float, target_y: float,
                           wheelbase: float, lookahead_m: float) -> float:
    """
    Steering angle that curves the vehicle onto a circular arc passing
    through the lookahead point. Returns steering angle in radians.
    """
    dx = target_x - vehicle_x
    dy = target_y - vehicle_y

    alpha = math.atan2(dy, dx) - vehicle_yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # wrap to [-pi, pi]

    return math.atan2(2 * wheelbase * math.sin(alpha), lookahead_m)


def speed_controller(current_speed: float, target_speed: float, kp: float = 0.5) -> float:
    """Simple proportional throttle controller. Returns throttle in [-1, 1]."""
    error = target_speed - current_speed
    return max(-1.0, min(1.0, kp * error))
