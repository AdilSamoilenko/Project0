import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.environment import (
    AtmosphereModel,
    ConstantWind,
    EnvironmentModel,
    gravity_acceleration,
)


def test_sea_level_atmosphere():
    atmosphere = AtmosphereModel()
    data = atmosphere.evaluate(0.0)

    assert abs(data["temperature_k"] - 288.15) < 1e-9
    assert abs(data["pressure_pa"] - 101325.0) < 1e-6
    assert abs(data["density_kg_m3"] - 1.225) < 0.002


def test_temperature_decreases_with_altitude():
    atmosphere = AtmosphereModel()

    sea_level = atmosphere.evaluate(0.0)
    high_altitude = atmosphere.evaluate(5000.0)

    assert high_altitude["temperature_k"] < sea_level["temperature_k"]
    assert high_altitude["pressure_pa"] < sea_level["pressure_pa"]
    assert high_altitude["density_kg_m3"] < sea_level["density_kg_m3"]


def test_wind_subtraction():
    environment = EnvironmentModel(
        wind=ConstantWind(
            np.array([10.0, 0.0, 0.0])
        )
    )

    state = environment.evaluate(
        time_s=0.0,
        position_m=np.array([0.0, 0.0, 100.0]),
        velocity_m_s=np.array([100.0, 0.0, 0.0]),
    )

    assert np.allclose(
        state.air_relative_velocity_m_s,
        np.array([90.0, 0.0, 0.0]),
    )


def test_dynamic_pressure():
    environment = EnvironmentModel()

    state = environment.evaluate(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.array([100.0, 0.0, 0.0]),
    )

    expected = 0.5 * state.density_kg_m3 * 100.0 ** 2

    assert abs(state.dynamic_pressure_pa - expected) < 1e-9


def test_mach_number():
    environment = EnvironmentModel()

    state = environment.evaluate(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.array([340.0, 0.0, 0.0]),
    )

    assert abs(
        state.mach - 340.0 / state.speed_of_sound_m_s
    ) < 1e-12


def test_gravity_decreases_with_altitude():
    sea_level = np.linalg.norm(gravity_acceleration(0.0))
    high_altitude = np.linalg.norm(gravity_acceleration(10000.0))

    assert high_altitude < sea_level


def test_environment_validation():
    environment = EnvironmentModel()

    state = environment.evaluate(
        time_s=0.0,
        position_m=np.array([0.0, 0.0, 1000.0]),
        velocity_m_s=np.array([50.0, 0.0, 20.0]),
    )

    state.validate()


def main():
    tests = [
        test_sea_level_atmosphere,
        test_temperature_decreases_with_altitude,
        test_wind_subtraction,
        test_dynamic_pressure,
        test_mach_number,
        test_gravity_decreases_with_altitude,
        test_environment_validation,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print()
    print("PROJECT 0 ENVIRONMENT TEST SUITE PASSED")


if __name__ == "__main__":
    main()
