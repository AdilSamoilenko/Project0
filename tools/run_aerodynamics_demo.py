import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.aerodynamics import (
    AerodynamicCoefficients,
    AerodynamicModel,
    calculate_aerodynamic_forces,
)


def main():
    coefficients = AerodynamicCoefficients(
        drag=0.30,
        lift=0.10,
        side=0.02,
        roll_moment=0.01,
        pitch_moment=0.02,
        yaw_moment=0.01,
    )

    model = AerodynamicModel(
        reference_area_m2=0.01,
        reference_length_m=1.0,
        coefficients=coefficients,
    )

    state = model.build_state(
        airspeed_m_s=120.0,
        mach=0.36,
        angle_of_attack_rad=0.05,
        sideslip_rad=0.01,
        dynamic_pressure_pa=7000.0,
    )

    forces = calculate_aerodynamic_forces(
        dynamic_pressure_pa=state.dynamic_pressure_pa,
        reference_area_m2=model.reference_area_m2,
        reference_length_m=model.reference_length_m,
        coefficients=model.coefficients,
    )

    print("PROJECT 0 AERODYNAMICS DEMONSTRATION")
    print("===================================")
    print(f"Airspeed:              {state.airspeed_m_s:.2f} m/s")
    print(f"Mach:                  {state.mach:.4f}")
    print(f"Angle of attack:       {state.angle_of_attack_rad:.5f} rad")
    print(f"Sideslip:              {state.sideslip_rad:.5f} rad")
    print(f"Dynamic pressure:      {state.dynamic_pressure_pa:.2f} Pa")
    print(
        "Force body:            "
        f"[{forces.force_body_n[0]:.3f}, "
        f"{forces.force_body_n[1]:.3f}, "
        f"{forces.force_body_n[2]:.3f}] N"
    )
    print(
        "Moment body:           "
        f"[{forces.moment_body_nm[0]:.3f}, "
        f"{forces.moment_body_nm[1]:.3f}, "
        f"{forces.moment_body_nm[2]:.3f}] N m"
    )
    print()
    print("AERODYNAMICS PIPELINE PASSED")


if __name__ == "__main__":
    main()
