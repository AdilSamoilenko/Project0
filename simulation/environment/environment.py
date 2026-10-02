import numpy as np

from .atmosphere import AtmosphereModel
from .wind import ConstantWind
from .gravity import gravity_acceleration
from .conditions import build_environment_state


class EnvironmentModel:
    def __init__(
        self,
        atmosphere=None,
        wind=None,
        gravity_model=gravity_acceleration,
    ):
        self.atmosphere = atmosphere or AtmosphereModel()
        self.wind = wind or ConstantWind(
            np.zeros(3, dtype=float)
        )
        self.gravity_model = gravity_model

    def evaluate(
        self,
        time_s,
        position_m,
        velocity_m_s,
    ):
        position = np.asarray(position_m, dtype=float)
        velocity = np.asarray(velocity_m_s, dtype=float)

        if position.shape != (3,):
            raise ValueError("Position must contain three components.")

        if velocity.shape != (3,):
            raise ValueError("Velocity must contain three components.")

        altitude = max(0.0, float(position[2]))

        return build_environment_state(
            altitude_m=altitude,
            ground_velocity_m_s=velocity,
            atmosphere=self.atmosphere,
            wind_model=self.wind,
            time_s=time_s,
            position_m=position,
            gravity_model=self.gravity_model,
        )
