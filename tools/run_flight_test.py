from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.core.flight import FlightParameters
from simulation.core.simulator import Simulator
from simulation.analysis.metrics import peak_altitude, peak_velocity
from simulation.analysis.plot import plot_altitude, plot_velocity
from telemetry.recorder import TelemetryRecorder


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
