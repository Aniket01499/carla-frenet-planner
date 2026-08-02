"""
vehicle_dynamics.py

Standalone (no CARLA dependency) implementation of the Nonlinear Dynamic
Single-Track (Bicycle) Model with a Fiala tire model, ported directly from
the thesis's MATLAB prediction function (update_veh_dynamics.m) and the
physical parameters in Table 6.2 (D-segment sedan, Ego/Target vehicle).

State vector: [X, Y, psi, v, r, beta]
  X, Y   -- global position (m)
  psi    -- yaw angle (rad)
  v      -- longitudinal speed (m/s)
  r      -- yaw rate (rad/s)
  beta   -- sideslip angle (rad)

Input: throttle in [-1, 1] (mapped linearly to longitudinal accel),
       steering delta (rad)

Deliberately has zero CARLA dependency: this is a pure function of
(state, input, dt) -> new_state, so it can be unit tested and sanity
checked (straight line under zero steering, turning under steering,
decelerating under braking) without ever launching the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VehicleParams:
    """Table 6.2 of the thesis: D-segment sedan."""
    mass: float = 1600.0        # kg
    Iz: float = 2500.0          # kg*m^2, yaw moment of inertia
    lf: float = 1.35            # m, CG to front axle
    lr: float = 1.35            # m, CG to rear axle
    csf: float = 80_000.0       # N/rad, front cornering stiffness
    csr: float = 100_000.0      # N/rad, rear cornering stiffness
    mu: float = 0.9             # tire-road friction coefficient

    # NOT confirmed from the thesis excerpts I have access to -- these are
    # reasonable placeholders (max braking ~ mu*g). Swap in your thesis's
    # actual a_long_max/a_long_min/actuation limits if you have them
    # recorded elsewhere (they'd be near Table 6.5, Controller Actuation
    # Constraints).
    a_long_max: float = 3.0     # m/s^2, max acceleration
    a_long_min: float = -8.83   # m/s^2, max braking (~ -mu*g)
    max_throttle: float = 1.0
    max_brake: float = -1.0


@dataclass
class BicycleState:
    X: float = 0.0
    Y: float = 0.0
    psi: float = 0.0    # yaw angle, rad
    v: float = 0.0       # longitudinal speed, m/s
    r: float = 0.0       # yaw rate, rad/s
    beta: float = 0.0    # sideslip angle, rad


def _fiala_lateral_force(alpha: float, cs: float, fx: float, fz: float, mu: float) -> float:
    """
    Nonlinear Fiala tire model with friction-circle coupling to the
    longitudinal force already committed at this axle. Ported directly
    from the thesis's MATLAB implementation.
    """
    fy_max_sq = (mu * fz) ** 2 - fx ** 2
    # Guard against fx exceeding mu*fz (a physically infeasible braking
    # command) -- the MATLAB source doesn't clamp this and would hit a
    # complex sqrt; clamped to 0 here instead (no lateral capacity left).
    fy_max = math.sqrt(max(fy_max_sq, 0.0))

    if fy_max == 0.0:
        return 0.0

    alpha_slip = math.atan((3 * fy_max) / (2 * cs))

    if abs(alpha) < alpha_slip:
        tan_a = math.tan(alpha)
        fy_raw = (
            -2 * cs * tan_a
            + ((2 * cs) ** 2 / (3 * fy_max)) * abs(tan_a) * tan_a
            - ((2 * cs) ** 3 / (27 * fy_max ** 2)) * tan_a ** 3
        )
        return -fy_raw  # sign inverted to match ISO convention, per thesis code
    else:
        return fy_max * math.copysign(1.0, alpha)


def step(state: BicycleState, throttle: float, steering: float,
         dt: float, params: VehicleParams) -> BicycleState:
    """One Euler-integration step. throttle in [-1, 1], steering in radians."""
    throttle = max(params.max_brake, min(params.max_throttle, throttle))

    ax = throttle * params.a_long_max if throttle >= 0 else throttle * abs(params.a_long_min)
    fx_total = params.mass * ax
    fx_f = fx_total * 0.5
    fx_r = fx_total * 0.5

    g = 9.81
    fz_f = (params.mass * g * params.lr) / (params.lf + params.lr)
    fz_r = (params.mass * g * params.lf) / (params.lf + params.lr)

    v_safe = max(state.v, 1.0)
    alpha_f = steering - state.beta - (params.lf * state.r) / v_safe
    alpha_r = -state.beta + (params.lr * state.r) / v_safe

    fyf = _fiala_lateral_force(alpha_f, params.csf, fx_f, fz_f, params.mu)
    fyr = _fiala_lateral_force(alpha_r, params.csr, fx_r, fz_r, params.mu)

    return BicycleState(
        X=state.X + state.v * math.cos(state.psi + state.beta) * dt,
        Y=state.Y + state.v * math.sin(state.psi + state.beta) * dt,
        psi=state.psi + state.r * dt,
        v=max(0.0, state.v + ax * dt),
        r=state.r + ((params.lf * fyf - params.lr * fyr) / params.Iz) * dt,
        beta=state.beta + ((fyf + fyr) / (params.mass * v_safe) - state.r) * dt,
    )


if __name__ == "__main__":
    # Sanity check, no CARLA required: straight line, then a turn, then braking.
    params = VehicleParams()
    state = BicycleState(v=15.0)

    print("Straight, zero steering, 1s:")
    for _ in range(50):
        state = step(state, throttle=0.3, steering=0.0, dt=0.02, params=params)
    print(f"  X={state.X:.2f} Y={state.Y:.2f} psi={math.degrees(state.psi):.2f}deg v={state.v:.2f}")

    print("Constant steering (0.1 rad), 2s:")
    for _ in range(100):
        state = step(state, throttle=0.1, steering=0.1, dt=0.02, params=params)
    print(f"  X={state.X:.2f} Y={state.Y:.2f} psi={math.degrees(state.psi):.2f}deg "
          f"v={state.v:.2f} beta={math.degrees(state.beta):.2f}deg")

    print("Hard braking, 1s:")
    for _ in range(50):
        state = step(state, throttle=-1.0, steering=0.0, dt=0.02, params=params)
    print(f"  v={state.v:.2f} (should be decreasing toward 0)")
