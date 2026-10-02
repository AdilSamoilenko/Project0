from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from simulation.models.aerodynamics import calculate_aero_state
from simulation.models.gravity import gravity_vector


@dataclass
class FlightParameters:
    mass_kg: float
    reference_area_m2: float


@dataclass
class FlightSample:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    altitude: float
    speed: float
    mach: float
    dynamic_pressure: float


class FlightModel:
    def __init__(self, parameters: FlightParameters):
        self.parameters = parameters

    def derivative(
        self,
        time: float,
        state: np.ndarray,
        force_function: Optional[Callable] = None,
    ) -> np.ndarray:

        position = state[0:3]
        velocity = state[3:6]

        altitude = max(float(position[2]), 0.0)

        gravity = gravity_vector(altitude)

        if force_function is None:
            external_force = np.zeros(3, dtype=float)
        else:
            external_force = np.asarray(
                force_function(time, state),
                dtype=float,
            )

        acceleration = (
            external_force / self.parameters.mass_kg
            + gravity
        )

        derivative = np.zeros(6, dtype=float)
        derivative[0:3] = velocity
        derivative[3:6] = acceleration

        return derivative

    def sample(
        self,
        time: float,
        state: np.ndarray,
        acceleration: np.ndarray,
    ) -> FlightSample:

        position = state[0:3]
        velocity = state[3:6]

        altitude = float(position[2])
        speed = float(np.linalg.norm(velocity))

        aero = calculate_aero_state(
            altitude_m=max(altitude, 0.0),
            velocity_vector=velocity,
        )

        return FlightSample(
            time=time,
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            altitude=altitude,
            speed=speed,
            mach=aero.mach,
            dynamic_pressure=aero.dynamic_pressure_pa,
        )
