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


def test_zero_coefficients():
    coefficients = AerodynamicCoefficients()

    result = calculate_aerodynamic_forces(
        dynamic_pressure_pa=1000.0,
        reference_area_m2=0.01,
        reference_length_m=1.0,
        coefficients=coefficients,
    )

    assert np.allclose(result.force_body_n, np.zeros(3))
    assert np.allclose(result.moment_body_nm, np.zeros(3))


def test_force_scaling():
    coefficients = AerodynamicCoefficients(
        drag=0.5,
        lift=0.2,
        side=0.1,
    )

    result = calculate_aerodynamic_forces(
        dynamic_pressure_pa=2000.0,
        reference_area_m2=0.5,
        reference_length_m=2.0,
        coefficients=coefficients,
    )

    assert np.allclose(
        result.force_body_n,
        np.array([-500.0, 100.0, -200.0]),
    )


def test_moment_scaling():
    coefficients = AerodynamicCoefficients(
        roll_moment=0.1,
        pitch_moment=0.2,
        yaw_moment=0.3,
    )

    result = calculate_aerodynamic_forces(
        dynamic_pressure_pa=1000.0,
        reference_area_m2=0.5,
        reference_length_m=2.0,
        coefficients=coefficients,
    )

    assert np.allclose(
        result.moment_body_nm,
        np.array([100.0, 200.0, 300.0]),
    )


def test_aerodynamic_state():
    model = AerodynamicModel(
        reference_area_m2=0.01,
        reference_length_m=1.0,
    )

    state = model.build_state(
        airspeed_m_s=100.0,
        mach=0.3,
        angle_of_attack_rad=0.05,
        sideslip_rad=0.01,
        dynamic_pressure_pa=5000.0,
    )

    assert state.airspeed_m_s == 100.0
    assert state.mach == 0.3
    assert state.dynamic_pressure_pa == 5000.0


def test_reference_validation():
    try:
        AerodynamicModel(
            reference_area_m2=0.0,
            reference_length_m=1.0,
        )
    except ValueError:
        return

    raise AssertionError("Invalid reference area was accepted.")


def test_coefficient_validation():
    coefficients = AerodynamicCoefficients(
        drag=0.3,
        lift=0.1,
        side=0.02,
    )

    coefficients.validate()


def main():
    tests = [
        test_zero_coefficients,
        test_force_scaling,
        test_moment_scaling,
        test_aerodynamic_state,
        test_reference_validation,
        test_coefficient_validation,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print()
    print("PROJECT 0 AERODYNAMICS TEST SUITE PASSED")


if __name__ == "__main__":
    main()
