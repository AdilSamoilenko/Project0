import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.core.state import VehicleState
from simulation.integration import IntegratedFlightModel


def main():
    initial = VehicleState(
        time_s=0.0,
        position_m=np.array([0.0, 0.0, 100.0]),
        velocity_m_s=np.array([20.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )

    model = IntegratedFlightModel()
    result = model.run(initial, duration_s=3.0, dt_s=0.01)
    final = result.samples[-1]

    print("PROJECT 0 INTEGRATED SIMULATION")
    print("===============================")
    print(f"Samples:            {len(result.samples)}")
    print(f"Final time:         {final['time_s']:.2f} s")
    print(f"Final altitude:     {final['altitude_m']:.3f} m")
    print(f"Final speed:        {final['speed_m_s']:.3f} m/s")
    print(f"Final airspeed:     {final['airspeed_m_s']:.3f} m/s")
    print(f"Final Mach:         {final['mach']:.5f}")
    print(f"Termination:        {result.termination_reason}")
    print()
    print("Illustrative software demonstration only; not a P0-X1 prediction.")


if __name__ == "__main__":
    main()
