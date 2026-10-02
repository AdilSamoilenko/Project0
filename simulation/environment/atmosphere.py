from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AtmosphereModel:
    sea_level_temperature_k: float = 288.15
    sea_level_pressure_pa: float = 101325.0
    temperature_lapse_rate_k_m: float = 0.0065
    gas_constant_j_kg_k: float = 287.05287
    gamma: float = 1.4

    def __post_init__(self):
        if self.sea_level_temperature_k <= 0.0:
            raise ValueError("Sea-level temperature must be positive.")
        if self.sea_level_pressure_pa <= 0.0:
            raise ValueError("Sea-level pressure must be positive.")
        if self.temperature_lapse_rate_k_m <= 0.0:
            raise ValueError("Temperature lapse rate must be positive.")
        if self.gas_constant_j_kg_k <= 0.0:
            raise ValueError("Gas constant must be positive.")
        if self.gamma <= 1.0:
            raise ValueError("Heat-capacity ratio must exceed 1.")

    def evaluate(self, altitude_m: float):
        altitude = max(0.0, float(altitude_m))

        if altitude <= 11000.0:
            temperature = (
                self.sea_level_temperature_k
                - self.temperature_lapse_rate_k_m * altitude
            )

            exponent = 9.80665 / (
                self.gas_constant_j_kg_k * self.temperature_lapse_rate_k_m
            )

            pressure = self.sea_level_pressure_pa * (
                temperature / self.sea_level_temperature_k
            ) ** exponent

        else:
            base_altitude = 11000.0
            base_temperature = (
                self.sea_level_temperature_k
                - self.temperature_lapse_rate_k_m * base_altitude
            )

            base_exponent = 9.80665 / (
                self.gas_constant_j_kg_k * self.temperature_lapse_rate_k_m
            )

            base_pressure = self.sea_level_pressure_pa * (
                base_temperature / self.sea_level_temperature_k
            ) ** base_exponent

            temperature = base_temperature
            pressure = base_pressure * math.exp(
                -9.80665
                * (altitude - base_altitude)
                / (self.gas_constant_j_kg_k * temperature)
            )

        density = pressure / (self.gas_constant_j_kg_k * temperature)
        speed_of_sound = math.sqrt(self.gamma * self.gas_constant_j_kg_k * temperature)

        return {
            "temperature_k": temperature,
            "pressure_pa": pressure,
            "density_kg_m3": density,
            "speed_of_sound_m_s": speed_of_sound,
        }


def standard_atmosphere(altitude_m: float):
    return AtmosphereModel().evaluate(altitude_m)
