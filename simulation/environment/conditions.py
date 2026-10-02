from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnvironmentState:
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    wind_velocity_m_s: np.ndarray
    air_relative_velocity_m_s: np.ndarray
    air_relative_speed_m_s: float
    mach: float
    dynamic_pressure_pa: float
    gravity_m_s2: np.ndarray

    def validate(self):
        if self.altitude_m < 0.0:
            raise ValueError("Altitude cannot be negative.")

        if self.temperature_k <= 0.0:
            raise ValueError("Temperature must be positive.")

        if self.pressure_pa <= 0.0:
            raise ValueError("Pressure must be positive.")

        if self.density_kg_m3 <= 0.0:
            raise ValueError("Density must be positive.")

        if self.speed_of_sound_m_s <= 0.0:
            raise ValueError("Speed of sound must be positive.")

        if self.wind_velocity_m_s.shape != (3,):
            raise ValueError("Wind velocity must have three components.")

        if self.air_relative_velocity_m_s.shape != (3,):
            raise ValueError("Air-relative velocity must have three components.")

        if self.gravity_m_s2.shape != (3,):
            raise ValueError("Gravity vector must have three components.")


def build_environment_state(
    altitude_m,
    ground_velocity_m_s,
    atmosphere,
    wind_model,
    time_s,
    position_m,
    gravity_model,
):
    atmosphere_data = atmosphere.evaluate(altitude_m)

    wind = np.asarray(
        wind_model.velocity_inertial(time_s, position_m),
        dtype=float,
    )

    ground_velocity = np.asarray(ground_velocity_m_s, dtype=float)

    air_relative_velocity = ground_velocity - wind
    air_relative_speed = float(np.linalg.norm(air_relative_velocity))

    speed_of_sound = atmosphere_data["speed_of_sound_m_s"]

    mach = air_relative_speed / speed_of_sound
    dynamic_pressure = (
        0.5
        * atmosphere_data["density_kg_m3"]
        * air_relative_speed ** 2
    )

    gravity = gravity_model(altitude_m)

    state = EnvironmentState(
        altitude_m=float(altitude_m),
        temperature_k=atmosphere_data["temperature_k"],
        pressure_pa=atmosphere_data["pressure_pa"],
        density_kg_m3=atmosphere_data["density_kg_m3"],
        speed_of_sound_m_s=speed_of_sound,
        wind_velocity_m_s=wind,
        air_relative_velocity_m_s=air_relative_velocity,
        air_relative_speed_m_s=air_relative_speed,
        mach=mach,
        dynamic_pressure_pa=dynamic_pressure,
        gravity_m_s2=np.asarray(gravity, dtype=float),
    )

    state.validate()
    return state
