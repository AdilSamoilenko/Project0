from dataclasses import dataclass
import math

from .coefficients import AerodynamicCoefficients


@dataclass(frozen=True)
class AerodynamicState:
    airspeed_m_s: float
    mach: float
    angle_of_attack_rad: float
    sideslip_rad: float
    dynamic_pressure_pa: float

    def validate(self):
        if self.airspeed_m_s < 0.0:
            raise ValueError("Airspeed cannot be negative.")

        if self.mach < 0.0:
            raise ValueError("Mach number cannot be negative.")

        if self.dynamic_pressure_pa < 0.0:
            raise ValueError("Dynamic pressure cannot be negative.")

        if not math.isfinite(self.angle_of_attack_rad):
            raise ValueError("Angle of attack must be finite.")

        if not math.isfinite(self.sideslip_rad):
            raise ValueError("Sideslip angle must be finite.")


class AerodynamicModel:
    def __init__(
        self,
        reference_area_m2,
        reference_length_m,
        coefficients=None,
    ):
        if reference_area_m2 <= 0.0:
            raise ValueError("Reference area must be positive.")

        if reference_length_m <= 0.0:
            raise ValueError("Reference length must be positive.")

        self.reference_area_m2 = float(reference_area_m2)
        self.reference_length_m = float(reference_length_m)
        self.coefficients = coefficients or AerodynamicCoefficients()

        self.coefficients.validate()

    def build_state(
        self,
        airspeed_m_s,
        mach,
        angle_of_attack_rad,
        sideslip_rad,
        dynamic_pressure_pa,
    ):
        state = AerodynamicState(
            airspeed_m_s=float(airspeed_m_s),
            mach=float(mach),
            angle_of_attack_rad=float(angle_of_attack_rad),
            sideslip_rad=float(sideslip_rad),
            dynamic_pressure_pa=float(dynamic_pressure_pa),
        )

        state.validate()
        return state
