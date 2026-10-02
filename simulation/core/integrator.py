from __future__ import annotations

from typing import Callable

import numpy as np

from simulation.core.dynamics import StateDerivative, add_derivative
from simulation.core.frames import normalise_quaternion
from simulation.core.state import VehicleState


DerivativeFunction = Callable[
    [float, VehicleState],
    StateDerivative,
]


def normalise_state(state: VehicleState) -> VehicleState:
    state.quaternion = normalise_quaternion(state.quaternion)
    return state


def rk4_step(
    state: VehicleState,
    dt_s: float,
    derivative_function: DerivativeFunction,
) -> VehicleState:

    if dt_s <= 0:
        raise ValueError("Integration timestep must be positive.")

    k1 = derivative_function(state.time_s, state)

    s2 = add_derivative(state, k1, dt_s / 2.0)
    k2 = derivative_function(s2.time_s, s2)

    s3 = add_derivative(state, k2, dt_s / 2.0)
    k3 = derivative_function(s3.time_s, s3)

    s4 = add_derivative(state, k3, dt_s)
    k4 = derivative_function(s4.time_s, s4)

    next_state = VehicleState(
        time_s=state.time_s + dt_s,
        position_m=(
            state.position_m
            + dt_s / 6.0
            * (
                k1.position_m_s
                + 2.0 * k2.position_m_s
                + 2.0 * k3.position_m_s
                + k4.position_m_s
            )
        ),
        velocity_m_s=(
            state.velocity_m_s
            + dt_s / 6.0
            * (
                k1.velocity_m_s2
                + 2.0 * k2.velocity_m_s2
                + 2.0 * k3.velocity_m_s2
                + k4.velocity_m_s2
            )
        ),
        quaternion=(
            state.quaternion
            + dt_s / 6.0
            * (
                k1.quaternion_s
                + 2.0 * k2.quaternion_s
                + 2.0 * k3.quaternion_s
                + k4.quaternion_s
            )
        ),
        angular_velocity_rad_s=(
            state.angular_velocity_rad_s
            + dt_s / 6.0
            * (
                k1.angular_velocity_rad_s2
                + 2.0 * k2.angular_velocity_rad_s2
                + 2.0 * k3.angular_velocity_rad_s2
                + k4.angular_velocity_rad_s2
            )
        ),
    )

    return normalise_state(next_state)
