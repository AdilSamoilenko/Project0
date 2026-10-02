from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from simulation.core.dynamics import (
    ForceMoment,
    RigidBodyParameters,
    rigid_body_derivative,
)
from simulation.core.integrator import rk4_step
from simulation.core.state import VehicleState


@dataclass
class FlightSample:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    quaternion: np.ndarray
    angular_velocity_rad_s: np.ndarray
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray
    speed_m_s: float
    altitude_m: float


ForceMomentFunction = Callable[[float, VehicleState], ForceMoment]


class SixDOFFlightModel:
    def __init__(
        self,
        parameters: RigidBodyParameters,
        force_moment_function: ForceMomentFunction,
        gravity_m_s2: float = 9.80665,
    ) -> None:

        self.parameters = parameters
        self.force_moment_function = force_moment_function
        self.gravity_m_s2 = float(gravity_m_s2)

        self.parameters.validate()

    def derivative(
        self,
        time_s: float,
        state: VehicleState,
    ):
        gravity = np.array(
            [0.0, 0.0, -self.gravity_m_s2],
            dtype=float,
        )

        return rigid_body_derivative(
            time_s=time_s,
            state=state,
            parameters=self.parameters,
            force_moment_model=self.force_moment_function,
            gravity_inertial_m_s2=gravity,
        )

    def step(
        self,
        state: VehicleState,
        dt_s: float,
    ) -> VehicleState:

        return rk4_step(
            state,
            dt_s,
            self.derivative,
        )

    def sample(self, state: VehicleState) -> FlightSample:
        force_moment = self.force_moment_function(
            state.time_s,
            state,
        )

        speed = float(np.linalg.norm(state.velocity_m_s))

        return FlightSample(
            time_s=state.time_s,
            position_m=state.position_m.copy(),
            velocity_m_s=state.velocity_m_s.copy(),
            quaternion=state.quaternion.copy(),
            angular_velocity_rad_s=state.angular_velocity_rad_s.copy(),
            force_body_n=force_moment.force_body_n.copy(),
            moment_body_nm=force_moment.moment_body_nm.copy(),
            speed_m_s=speed,
            altitude_m=float(state.position_m[2]),
        )


def run_6dof_simulation(
    initial_state: VehicleState,
    model: SixDOFFlightModel,
    duration_s: float,
    dt_s: float,
) -> list[FlightSample]:

    if duration_s <= 0:
        raise ValueError("Simulation duration must be positive.")

    if dt_s <= 0:
        raise ValueError("Simulation timestep must be positive.")

    samples = [model.sample(initial_state)]

    state = initial_state.copy()
    elapsed = 0.0

    while elapsed < duration_s - 1e-12:
        step = min(dt_s, duration_s - elapsed)
        state = model.step(state, step)
        samples.append(model.sample(state))
        elapsed += step

    return samples
