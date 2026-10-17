import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.core.state import VehicleState
from simulation.integration import IntegratedFlightModel
from simulation.environment.environment import EnvironmentModel
from simulation.aerodynamics import AerodynamicCoefficients


class TestEnvironment:
    def evaluate(self, time_s, position_m, velocity_m_s):
        return {
            "wind_velocity_m_s": np.zeros(3),
            "density_kg_m3": 1.225,
            "speed_of_sound_m_s": 340.0,
            "dynamic_pressure_pa": 0.5 * 1.225 * float(
                np.dot(velocity_m_s, velocity_m_s)
            ),
            "gravity_vector_m_s2": np.array([0.0, 0.0, -9.80665]),
        }


def initial_state():
    return VehicleState(
        time_s=0.0,
        position_m=np.array([0.0, 0.0, 100.0]),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )


def test_zero_aero_free_fall():
    model = IntegratedFlightModel(
        environment=TestEnvironment(),
        coefficients=AerodynamicCoefficients(),
    )
    result = model.run(initial_state(), duration_s=1.0, dt_s=0.01)
    final = result.samples[-1]

    assert abs(final["position_m"][2] - 95.096675) < 0.02
    assert abs(final["velocity_m_s"][2] + 9.80665) < 0.02
    assert result.termination_reason == "duration_reached"


def test_quaternion_normalisation():
    model = IntegratedFlightModel(environment=TestEnvironment())
    result = model.run(initial_state(), duration_s=0.5, dt_s=0.01)

    for sample in result.samples:
        assert abs(np.linalg.norm(sample["quaternion"]) - 1.0) < 1e-10


def test_aerodynamics_changes_motion():
    model = IntegratedFlightModel(
        environment=TestEnvironment(),
        coefficients=AerodynamicCoefficients(drag=0.3),
    )
    state = initial_state()
    state.velocity_m_s = np.array([20.0, 0.0, 0.0])
    result = model.run(state, duration_s=0.2, dt_s=0.005)

    assert result.samples[-1]["velocity_m_s"][0] < 20.0


def test_invalid_mass_rejected():
    try:
        IntegratedFlightModel(mass_kg=0.0)
    except ValueError:
        return
    raise AssertionError("Invalid mass was accepted.")


def main():
    tests = [
        test_zero_aero_free_fall,
        test_quaternion_normalisation,
        test_aerodynamics_changes_motion,
        test_invalid_mass_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("\nPROJECT 0 INTEGRATION TEST SUITE PASSED")


if __name__ == "__main__":
    main()
