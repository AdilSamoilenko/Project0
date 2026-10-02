from dataclasses import dataclass
import numpy as np

from .coefficients import AerodynamicCoefficients


@dataclass(frozen=True)
class AerodynamicForces:
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray

    def validate(self):
        if self.force_body_n.shape != (3,):
            raise ValueError("Aerodynamic force must contain three components.")

        if self.moment_body_nm.shape != (3,):
            raise ValueError("Aerodynamic moment must contain three components.")


def calculate_aerodynamic_forces(
    dynamic_pressure_pa,
    reference_area_m2,
    reference_length_m,
    coefficients,
):
    if dynamic_pressure_pa < 0.0:
        raise ValueError("Dynamic pressure cannot be negative.")

    if reference_area_m2 <= 0.0:
        raise ValueError("Reference area must be positive.")

    if reference_length_m <= 0.0:
        raise ValueError("Reference length must be positive.")

    if not isinstance(coefficients, AerodynamicCoefficients):
        raise TypeError(
            "coefficients must be an AerodynamicCoefficients instance."
        )

    q = float(dynamic_pressure_pa)
    area = float(reference_area_m2)
    length = float(reference_length_m)

    force_scale = q * area
    moment_scale = q * area * length

    drag = force_scale * coefficients.drag
    side = force_scale * coefficients.side
    lift = force_scale * coefficients.lift

    roll = moment_scale * coefficients.roll_moment
    pitch = moment_scale * coefficients.pitch_moment
    yaw = moment_scale * coefficients.yaw_moment

    result = AerodynamicForces(
        force_body_n=np.array(
            [-drag, side, -lift],
            dtype=float,
        ),
        moment_body_nm=np.array(
            [roll, pitch, yaw],
            dtype=float,
        ),
    )

    result.validate()
    return result
