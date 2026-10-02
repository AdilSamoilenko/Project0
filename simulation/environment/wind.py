from dataclasses import dataclass
import numpy as np


class WindModel:
    def velocity_inertial(self, time_s: float, position_m: np.ndarray):
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantWind(WindModel):
    velocity_m_s: np.ndarray

    def __post_init__(self):
        velocity = np.asarray(self.velocity_m_s, dtype=float)

        if velocity.shape != (3,):
            raise ValueError("Wind velocity must contain three components.")

        object.__setattr__(self, "velocity_m_s", velocity)

    def velocity_inertial(self, time_s: float, position_m: np.ndarray):
        return self.velocity_m_s.copy()
