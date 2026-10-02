from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.core.frames import (
    normalise_quaternion,
    quaternion_to_rotation_matrix,
)
from simulation.models.atmosphere import standard_atmosphere
from simulation.models.aerodynamics import calculate_aero_state
from simulation.models.gravity import gravity_magnitude


def check_quaternion():

    q = np.array(
        [2.0, 0.0, 0.0, 0.0],
        dtype=float,
    )

    result = normalise_quaternion(q)

    assert np.allclose(
        result,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def check_rotation_matrix():

    q = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=float,
    )

    rotation = quaternion_to_rotation_matrix(q)

    assert np.allclose(
        rotation,
        np.eye(3),
    )


def check_atmosphere():

    atmosphere = standard_atmosphere(0.0)

    assert abs(
        atmosphere.temperature_k - 288.15
    ) < 1e-6

    assert abs(
        atmosphere.pressure_pa - 101325.0
    ) < 1e-3

    assert atmosphere.density_kg_m3 > 1.0

    assert atmosphere.speed_of_sound_m_s > 300.0


def check_aerodynamics():

    aero = calculate_aero_state(
        altitude_m=0.0,
        velocity_vector=np.array(
            [100.0, 0.0, 0.0],
            dtype=float,
        ),
    )

    assert abs(aero.speed_m_s - 100.0) < 1e-9
    assert aero.mach > 0.0
    assert aero.dynamic_pressure_pa > 0.0


def check_gravity():

    gravity = gravity_magnitude(0.0)

    assert abs(
        gravity - 9.80665
    ) < 1e-6


def main():

    tests = [
        ("Quaternion normalisation", check_quaternion),
        ("Quaternion rotation matrix", check_rotation_matrix),
        ("Standard atmosphere", check_atmosphere),
        ("Aerodynamic state", check_aerodynamics),
        ("Gravity model", check_gravity),
    ]

    print()
    print("PROJECT 0 CORE TESTS")
    print("====================")

    for name, test in tests:

        test()

        print(f"PASS  {name}")

    print()
    print("ALL CORE TESTS PASSED")


if __name__ == "__main__":
    main()
