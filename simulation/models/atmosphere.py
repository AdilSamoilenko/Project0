from dataclasses import dataclass
import math


@dataclass
class Atmosphere:
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float


# Standard atmosphere constants
SEA_LEVEL_TEMPERATURE_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0
TEMPERATURE_LAPSE_RATE_K_M = 0.0065
GAS_CONSTANT_AIR = 287.05287
GRAVITY = 9.80665
GAMMA_AIR = 1.4


def standard_atmosphere(altitude_m: float) -> Atmosphere:
    """
    Simplified standard atmosphere model.

    Valid for the initial PROJECT 0 simulation framework.
    The model uses:
      - troposphere from 0 to 11 km
      - isothermal layer above 11 km

    SI units are used throughout.
    """

    altitude_m = max(float(altitude_m), 0.0)

    if altitude_m <= 11000.0:

        temperature = (
            SEA_LEVEL_TEMPERATURE_K
            - TEMPERATURE_LAPSE_RATE_K_M * altitude_m
        )

        pressure = (
            SEA_LEVEL_PRESSURE_PA
            * (
                temperature
                / SEA_LEVEL_TEMPERATURE_K
            )
            ** (
                GRAVITY
                / (
                    GAS_CONSTANT_AIR
                    * TEMPERATURE_LAPSE_RATE_K_M
                )
            )
        )

    else:

        temperature = (
            SEA_LEVEL_TEMPERATURE_K
            - TEMPERATURE_LAPSE_RATE_K_M * 11000.0
        )

        pressure_11km = (
            SEA_LEVEL_PRESSURE_PA
            * (
                temperature
                / SEA_LEVEL_TEMPERATURE_K
            )
            ** (
                GRAVITY
                / (
                    GAS_CONSTANT_AIR
                    * TEMPERATURE_LAPSE_RATE_K_M
                )
            )
        )

        pressure = (
            pressure_11km
            * math.exp(
                -GRAVITY
                * (altitude_m - 11000.0)
                / (
                    GAS_CONSTANT_AIR
                    * temperature
                )
            )
        )

    density = pressure / (
        GAS_CONSTANT_AIR
        * temperature
    )

    speed_of_sound = math.sqrt(
        GAMMA_AIR
        * GAS_CONSTANT_AIR
        * temperature
    )

    return Atmosphere(
        temperature_k=temperature,
        pressure_pa=pressure,
        density_kg_m3=density,
        speed_of_sound_m_s=speed_of_sound,
    )
