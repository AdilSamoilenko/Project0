from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from simulation.core.frames import quaternion_derivative
from simulation.core.state import VehicleState


@dataclass
class RigidBodyParameters:
    mass_kg: float
    inertia_kg_m2: np.ndarray

    def validate(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("Mass must be positive.")

        inertia = np.asarray(self.inertia_kg_m2, dtype=float)

        if inertia.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3 by 3.")

        if not np.allclose(inertia, inertia.T, atol=1e-12):
            raise ValueError("Inertia tensor must be symmetric.")

        eigenvalues = np.linalg.eigvalsh(inertia)

        if np.any(eigenvalues <= 0):
            raise ValueError("Inertia tensor must be positive definite.")


@dataclass
class ForceMoment:
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray

    def __post_init__(self) -> None:
        self.force_body_n = np.asarray(self.force_body_n, dtype=float)
        self.moment_body_nm = np.asarray(self.moment_body_nm, dtype=float)


@dataclass
class StateDerivative:
    position_m_s: np.ndarray
    velocity_m_s2: np.ndarray
    quaternion_s: np.ndarray
    angular_velocity_rad_s2: np.ndarray


ForceMomentModel = Callable[[float, VehicleState], ForceMoment]


def rigid_body_derivative(
    time_s: float,
    state: VehicleState,
    parameters: RigidBodyParameters,
    force_moment_model: ForceMomentModel,
    gravity_inertial_m_s2: np.ndarray,
) -> StateDerivative:

    parameters.validate()
    state.validate()

    gravity = np.asarray(gravity_inertial_m_s2, dtype=float)

    if gravity.shape != (3,):
        raise ValueError("Gravity must contain three components.")

    applied = force_moment_model(time_s, state)

    if applied.force_body_n.shape != (3,):
        raise ValueError("Body force must contain three components.")

    if applied.moment_body_nm.shape != (3,):
        raise ValueError("Body moment must contain three components.")

    from simulation.core.frames import body_to_inertial, inertial_to_body

    force_inertial_n = body_to_inertial(
        applied.force_body_n,
        state.quaternion,
    )

    acceleration_inertial_m_s2 = (
        force_inertial_n / parameters.mass_kg
        + gravity
    )

    angular_velocity = state.angular_velocity_rad_s

    inertia = parameters.inertia_kg_m2

    rotational_rhs = (
        applied.moment_body_nm
        - np.cross(
            angular_velocity,
            inertia @ angular_velocity,
        )
    )

    angular_acceleration = np.linalg.solve(
        inertia,
        rotational_rhs,
    )

    return StateDerivative(
        position_m_s=state.velocity_m_s.copy(),
        velocity_m_s2=acceleration_inertial_m_s2,
        quaternion_s=quaternion_derivative(
            state.quaternion,
            angular_velocity,
        ),
        angular_velocity_rad_s2=angular_acceleration,
    )


def add_derivative(
    state: VehicleState,
    derivative: StateDerivative,
    scale: float,
) -> VehicleState:

    return VehicleState(
        time_s=state.time_s + scale,
        position_m=state.position_m + derivative.position_m_s * scale,
        velocity_m_s=state.velocity_m_s + derivative.velocity_m_s2 * scale,
        quaternion=state.quaternion + derivative.quaternion_s * scale,
        angular_velocity_rad_s=(
            state.angular_velocity_rad_s
            + derivative.angular_velocity_rad_s2 * scale
        ),
    )
