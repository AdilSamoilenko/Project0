from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import numpy as np

from simulation.core.dynamics import ForceMoment
from simulation.core.flight import SixDOFFlightModel, run_6dof_simulation
from simulation.core.state import VehicleState
from simulation.models.vehicle import VehicleModel


def force_moment_model(time_s, state):
    return ForceMoment(
        force_body_n=np.array([0.0, 0.0, 120.0]),
        moment_body_nm=np.zeros(3),
    )


def main():
    vehicle = VehicleModel(
        mass_kg=10.0,
        inertia_xx_kg_m2=1.0,
        inertia_yy_kg_m2=1.0,
        inertia_zz_kg_m2=0.2,
        reference_area_m2=0.01,
    )

    vehicle.validate()

    initial_state = VehicleState(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )

    model = SixDOFFlightModel(
        parameters=vehicle.rigid_body_parameters(),
        force_moment_function=force_moment_model,
    )

    samples = run_6dof_simulation(
        initial_state=initial_state,
        model=model,
        duration_s=5.0,
        dt_s=0.01,
    )

    output_directory = ROOT / "simulation" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "sixdof_demo.csv"

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "time_s",
            "x_m",
            "y_m",
            "z_m",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
            "q0",
            "q1",
            "q2",
            "q3",
            "wx_rad_s",
            "wy_rad_s",
            "wz_rad_s",
            "speed_m_s",
            "altitude_m",
        ])

        for sample in samples:
            writer.writerow([
                sample.time_s,
                *sample.position_m,
                *sample.velocity_m_s,
                *sample.quaternion,
                *sample.angular_velocity_rad_s,
                sample.speed_m_s,
                sample.altitude_m,
            ])

    peak_altitude = max(sample.altitude_m for sample in samples)
    peak_speed = max(sample.speed_m_s for sample in samples)

    print("PROJECT 0 6-DOF DEMONSTRATION")
    print("=============================")
    print(f"Samples:       {len(samples)}")
    print(f"Peak altitude: {peak_altitude:.3f} m")
    print(f"Peak speed:    {peak_speed:.3f} m/s")
    print(f"CSV:           {output_file}")
    print()
    print("6-DOF PIPELINE PASSED")


if __name__ == "__main__":
    main()
