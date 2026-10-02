from typing import Callable, Optional

import numpy as np

from simulation.core.flight import FlightModel, FlightParameters, FlightSample
from simulation.core.integrator import RK4


class Simulator:
    def __init__(self, parameters: FlightParameters):
        self.model = FlightModel(parameters)

    def run(
        self,
        duration_s: float,
        timestep_s: float,
        initial_state: Optional[np.ndarray] = None,
        force_function: Optional[Callable] = None,
    ) -> list[FlightSample]:

        if initial_state is None:
            state = np.zeros(6, dtype=float)
        else:
            state = np.asarray(initial_state, dtype=float).copy()

        samples = []

        time = 0.0

        while time <= duration_s + 1e-12:

            derivative = self.model.derivative(
                time,
                state,
                force_function,
            )

            acceleration = derivative[3:6]

            samples.append(
                self.model.sample(
                    time,
                    state,
                    acceleration,
                )
            )

            state = RK4.step(
                lambda t, y: self.model.derivative(
                    t,
                    y,
                    force_function,
                ),
                time,
                state,
                timestep_s,
            )

            time += timestep_s

        return samples
