import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.environment import EnvironmentModel, ConstantWind


def main():
    environment = EnvironmentModel(
        wind=ConstantWind(
            np.array([8.0, 2.0, 0.0])
        )
    )

    position = np.array([0.0, 0.0, 1500.0])
    velocity = np.array([120.0, 0.0, 15.0])

    state = environment.evaluate(
        time_s=12.0,
        position_m=position,
        velocity_m_s=velocity,
    )

    print("PROJECT 0 ENVIRONMENT DEMONSTRATION")
    print("===================================")
    print(f"Altitude:              {state.altitude_m:.2f} m")
    print(f"Temperature:           {state.temperature_k:.2f} K")
    print(f"Pressure:              {state.pressure_pa:.2f} Pa")
    print(f"Density:               {state.density_kg_m3:.5f} kg/m^3")
    print(f"Speed of sound:        {state.speed_of_sound_m_s:.2f} m/s")
    print(
        "Wind:                  "
        f"[{state.wind_velocity_m_s[0]:.2f}, "
        f"{state.wind_velocity_m_s[1]:.2f}, "
        f"{state.wind_velocity_m_s[2]:.2f}] m/s"
    )
    print(
        "Air-relative velocity: "
        f"[{state.air_relative_velocity_m_s[0]:.2f}, "
        f"{state.air_relative_velocity_m_s[1]:.2f}, "
        f"{state.air_relative_velocity_m_s[2]:.2f}] m/s"
    )
    print(f"Air-relative speed:    {state.air_relative_speed_m_s:.2f} m/s")
    print(f"Mach:                  {state.mach:.4f}")
    print(f"Dynamic pressure:      {state.dynamic_pressure_pa:.2f} Pa")
    print(
        "Gravity:               "
        f"{np.linalg.norm(state.gravity_m_s2):.5f} m/s^2"
    )
    print()
    print("ENVIRONMENT PIPELINE PASSED")


if __name__ == "__main__":
    main()
