from pathlib import Path

ROOT = Path(__file__).resolve().parent

files = {
    "simulation/core/flight.py": '''
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
            dynamic_pressure=aero.dynamic_pressure,
        )
''',

    "simulation/core/simulator.py": '''
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
''',

    "telemetry/recorder.py": '''
import csv
from pathlib import Path

from simulation.core.flight import FlightSample


class TelemetryRecorder:

    @staticmethod
    def samples_to_rows(samples: list[FlightSample]):

        rows = []

        for sample in samples:

            rows.append(
                {
                    "time_s": sample.time,
                    "x_m": sample.position[0],
                    "y_m": sample.position[1],
                    "z_m": sample.position[2],
                    "vx_m_s": sample.velocity[0],
                    "vy_m_s": sample.velocity[1],
                    "vz_m_s": sample.velocity[2],
                    "ax_m_s2": sample.acceleration[0],
                    "ay_m_s2": sample.acceleration[1],
                    "az_m_s2": sample.acceleration[2],
                    "altitude_m": sample.altitude,
                    "speed_m_s": sample.speed,
                    "mach": sample.mach,
                    "dynamic_pressure_pa": sample.dynamic_pressure,
                }
            )

        return rows

    @staticmethod
    def save_csv(samples: list[FlightSample], path: str | Path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = TelemetryRecorder.samples_to_rows(samples)

        if not rows:
            return

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)
''',

    "simulation/analysis/plot.py": '''
from pathlib import Path

import matplotlib.pyplot as plt


def plot_altitude(samples, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = [sample.time for sample in samples]
    altitude = [sample.altitude for sample in samples]

    plt.figure()
    plt.plot(time, altitude)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("PROJECT 0 Flight Test: Altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_velocity(samples, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = [sample.time for sample in samples]
    speed = [sample.speed for sample in samples]

    plt.figure()
    plt.plot(time, speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("PROJECT 0 Flight Test: Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
''',

    "tools/run_flight_test.py": '''
from pathlib import Path

import numpy as np

from simulation.core.flight import FlightParameters
from simulation.core.simulator import Simulator
from simulation.analysis.metrics import peak_altitude, peak_velocity
from simulation.analysis.plot import plot_altitude, plot_velocity
from telemetry.recorder import TelemetryRecorder


ROOT = Path(__file__).resolve().parents[1]


def external_force(time_s, state):

    # Generic software validation force.
    # This is not a propulsion model.

    if time_s < 2.0:
        return np.array(
            [0.0, 0.0, 150.0],
            dtype=float,
        )

    return np.zeros(3, dtype=float)


def main():

    parameters = FlightParameters(
        mass_kg=10.0,
        reference_area_m2=0.01,
    )

    simulator = Simulator(parameters)

    samples = simulator.run(
        duration_s=10.0,
        timestep_s=0.01,
        force_function=external_force,
    )

    csv_file = ROOT / "simulation" / "output" / "flight_test.csv"
    altitude_plot = ROOT / "simulation" / "output" / "altitude.png"
    velocity_plot = ROOT / "simulation" / "output" / "velocity.png"

    TelemetryRecorder.save_csv(
        samples,
        csv_file,
    )

    plot_altitude(
        samples,
        altitude_plot,
    )

    plot_velocity(
        samples,
        velocity_plot,
    )

    print()
    print("PROJECT 0 FLIGHT TEST")
    print("=====================")
    print(f"Samples:       {len(samples)}")
    print(f"Peak altitude: {peak_altitude(samples):.2f} m")
    print(f"Peak speed:    {peak_velocity(samples):.2f} m/s")
    print()
    print(f"CSV: {csv_file}")
    print()
    print("FLIGHT PIPELINE PASSED")


if __name__ == "__main__":
    main()
'''
}


for filename, content in files.items():

    path = ROOT / filename

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )

print("PROJECT 0 FLIGHT SOFTWARE INSTALLED")
print(f"Files installed: {len(files)}")
