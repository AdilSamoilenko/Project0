from dataclasses import dataclass
import numpy as np

from simulation.models.atmosphere import standard_atmosphere


@dataclass
class AeroState:
    altitude_m: float
    speed_m_s: float
    mach: float
    dynamic_pressure_pa: float
    density_kg_m3: float


def calculate_aero_state(
    altitude_m,
    velocity_vector,
):
    velocity_vector = np.asarray(
        velocity_vector,
        dtype=float,
    )

    speed = float(np.linalg.norm(velocity_vector))

    atmosphere = standard_atmosphere(altitude_m)

    if atmosphere.speed_of_sound_m_s > 0:
        mach = speed / atmosphere.speed_of_sound_m_s
    else:
        mach = 0.0

    dynamic_pressure = (
        0.5
        * atmosphere.density_kg_m3
        * speed ** 2
    )

    return AeroState(
        altitude_m=float(altitude_m),
        speed_m_s=speed,
        mach=mach,
        dynamic_pressure_pa=dynamic_pressure,
        density_kg_m3=atmosphere.density_kg_m3,
    )
