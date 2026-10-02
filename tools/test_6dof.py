from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from simulation.core.dynamics import ForceMoment, RigidBodyParameters
from simulation.core.frames import (
    body_to_inertial,
    inertial_to_body,
    normalise_quaternion,
    quaternion_multiply,
)
from simulation.core.flight import SixDOFFlightModel, run_6dof_simulation
from simulation.core.state import VehicleState


def zero_force_moment(time_s, state):
    return ForceMoment(
        force_body_n=np.zeros(3),
        moment_body_nm=np.zeros(3),
    )


def constant_force(time_s, state):
    return ForceMoment(
        force_body_n=np.array([0.0, 0.0, 10.0]),
        moment_body_nm=np.zeros(3),
    )


def test_quaternion_normalisation():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    result = normalise_quaternion(q)

    assert np.isclose(np.linalg.norm(result), 1.0)


def test_frame_round_trip():
    vector = np.array([1.2, -3.4, 5.6])
    quaternion = normalise_quaternion(
        np.array([0.9, 0.2, -0.1, 0.3])
    )

    inertial = body_to_inertial(vector, quaternion)
    recovered = inertial_to_body(inertial, quaternion)

    assert np.allclose(vector, recovered, atol=1e-10)


def test_free_fall():
    parameters = RigidBodyParameters(
        mass_kg=10.0,
        inertia_kg_m2=np.diag([1.0, 1.0, 1.0]),
    )

    initial = VehicleState(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )

    model = SixDOFFlightModel(
        parameters=parameters,
        force_moment_function=zero_force_moment,
    )

    samples = run_6dof_simulation(
        initial_state=initial,
        model=model,
        duration_s=2.0,
        dt_s=0.01,
    )

    final = samples[-1]

    expected_velocity = -9.80665 * 2.0
    expected_altitude = 0.5 * -9.80665 * 2.0 ** 2

    assert np.isclose(
        final.velocity_m_s[2],
        expected_velocity,
        atol=1e-8,
    )

    assert np.isclose(
        final.position_m[2],
        expected_altitude,
        atol=1e-8,
    )


def test_constant_force():
    parameters = RigidBodyParameters(
        mass_kg=10.0,
        inertia_kg_m2=np.diag([1.0, 1.0, 1.0]),
    )

    initial = VehicleState(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )

    model = SixDOFFlightModel(
        parameters=parameters,
        force_moment_function=constant_force,
    )

    samples = run_6dof_simulation(
        initial_state=initial,
        model=model,
        duration_s=1.0,
        dt_s=0.01,
    )

    final = samples[-1]

    expected_acceleration = 1.0 - 9.80665

    assert np.isclose(
        final.velocity_m_s[2],
        expected_acceleration,
        atol=1e-8,
    )


def test_quaternion_remains_normalised():
    parameters = RigidBodyParameters(
        mass_kg=10.0,
        inertia_kg_m2=np.diag([1.0, 2.0, 3.0]),
    )

    def constant_moment(time_s, state):
        return ForceMoment(
            force_body_n=np.zeros(3),
            moment_body_nm=np.array([0.1, 0.2, 0.3]),
        )

    initial = VehicleState(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.array([0.1, 0.2, 0.3]),
    )

    model = SixDOFFlightModel(
        parameters=parameters,
        force_moment_function=constant_moment,
    )

    samples = run_6dof_simulation(
        initial_state=initial,
        model=model,
        duration_s=2.0,
        dt_s=0.005,
    )

    for sample in samples:
        assert np.isclose(
            np.linalg.norm(sample.quaternion),
            1.0,
            atol=1e-10,
        )


def test_rigid_body_simulation_runs():
    parameters = RigidBodyParameters(
        mass_kg=10.0,
        inertia_kg_m2=np.diag([1.0, 1.0, 1.0]),
    )

    initial = VehicleState(
        time_s=0.0,
        position_m=np.zeros(3),
        velocity_m_s=np.array([5.0, 0.0, 10.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.array([0.1, 0.2, 0.3]),
    )

    model = SixDOFFlightModel(
        parameters=parameters,
        force_moment_function=zero_force_moment,
    )

    samples = run_6dof_simulation(
        initial_state=initial,
        model=model,
        duration_s=1.0,
        dt_s=0.01,
    )

    assert len(samples) == 101
    assert all(np.isfinite(sample.speed_m_s) for sample in samples)


def main():
    tests = [
        test_quaternion_normalisation,
        test_frame_round_trip,
        test_free_fall,
        test_constant_force,
        test_quaternion_remains_normalised,
        test_rigid_body_simulation_runs,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print()
    print("PROJECT 0 6-DOF TEST SUITE PASSED")


if __name__ == "__main__":
    main()
