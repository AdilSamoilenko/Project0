import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.core.state import VehicleState
from simulation.core.frames import quaternion_to_rotation_matrix
from simulation.integration import IntegratedFlightModel
from simulation.aerodynamics import AerodynamicCoefficients


class VerificationEnvironment:
    """Deterministic constant-density test environment."""

    def evaluate(self, time_s, position_m, velocity_m_s):
        velocity = np.asarray(velocity_m_s, dtype=float)
        density = 1.225
        return {
            "wind_velocity_m_s": np.zeros(3),
            "density_kg_m3": density,
            "speed_of_sound_m_s": 340.0,
            "dynamic_pressure_pa": 0.5 * density * float(
                np.dot(velocity, velocity)
            ),
            "gravity_vector_m_s2": np.array(
                [0.0, 0.0, -9.80665]
            ),
        }


def make_state(position=(0.0, 0.0, 100.0),
               velocity=(0.0, 0.0, 0.0),
               quaternion=(1.0, 0.0, 0.0, 0.0),
               omega=(0.0, 0.0, 0.0)):
    return VehicleState(
        time_s=0.0,
        position_m=np.array(position, dtype=float),
        velocity_m_s=np.array(velocity, dtype=float),
        quaternion=np.array(quaternion, dtype=float),
        angular_velocity_rad_s=np.array(omega, dtype=float),
    )


def model_with(coefficients, inertia=(0.1, 0.1, 0.1)):
    return IntegratedFlightModel(
        mass_kg=1.0,
        inertia_kg_m2=inertia,
        coefficients=coefficients,
        environment=VerificationEnvironment(),
    )


def check_free_fall():
    gravity = 9.80665
    duration = 1.0
    initial_z = 100.0
    initial_vz = 0.0

    model = model_with(AerodynamicCoefficients())
    result = model.run(
        make_state(),
        duration_s=duration,
        dt_s=0.02,
    )
    final = result.samples[-1]

    expected_z = initial_z + initial_vz * duration - (
        0.5 * gravity * duration**2
    )
    expected_vz = initial_vz - gravity * duration

    position_error = abs(final["position_m"][2] - expected_z)
    velocity_error = abs(final["velocity_m_s"][2] - expected_vz)

    assert position_error < 1e-8, position_error
    assert velocity_error < 1e-8, velocity_error

    return {
        "position_error_m": position_error,
        "velocity_error_m_s": velocity_error,
        "duration_s": duration,
    }


def check_frame_rotation():
    q = np.array([0.9238795325, 0.0, 0.3826834324, 0.0])
    q = q / np.linalg.norm(q)
    rotation = quaternion_to_rotation_matrix(q)

    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3))
    )
    determinant = float(np.linalg.det(rotation))

    assert orthogonality_error < 1e-10, orthogonality_error
    assert abs(determinant - 1.0) < 1e-10, determinant

    vector = np.array([3.0, -2.0, 5.0])
    round_trip_error = float(
        np.linalg.norm(rotation.T @ (rotation @ vector) - vector)
    )
    assert round_trip_error < 1e-10, round_trip_error

    return {
        "orthogonality_error": orthogonality_error,
        "determinant": determinant,
        "round_trip_error": round_trip_error,
    }


def run_drag_case(dt):
    model = model_with(
        AerodynamicCoefficients(drag=0.3)
    )
    result = model.run(
        make_state(velocity=(20.0, 0.0, 0.0)),
        duration_s=2.0,
        dt_s=dt,
    )
    final = result.samples[-1]
    return np.concatenate((
        final["position_m"],
        final["velocity_m_s"],
    ))


def check_timestep_convergence():
    reference = run_drag_case(0.000625)
    steps = [0.04, 0.02, 0.01, 0.005]
    errors = {}

    for dt in steps:
        state = run_drag_case(dt)
        errors[str(dt)] = float(np.linalg.norm(state - reference))

    assert errors["0.005"] < errors["0.04"], errors
    assert errors["0.005"] < 0.02, errors

    return {
        "reference_dt_s": 0.000625,
        "errors_against_fine_reference": errors,
        "interpretation": (
            "Errors are measured against a finer numerical run, "
            "not experimental data."
        ),
    }


def check_torque_free_rotation():
    inertia = np.array([0.1, 0.12, 0.15])
    omega0 = np.array([0.7, 1.1, 0.4])

    model = model_with(
        AerodynamicCoefficients(),
        inertia=tuple(inertia),
    )
    result = model.run(
        make_state(omega=omega0),
        duration_s=2.0,
        dt_s=0.001,
    )

    def invariants(omega):
        angular_momentum = inertia * omega
        energy = 0.5 * float(np.dot(inertia * omega, omega))
        momentum_norm = float(np.linalg.norm(angular_momentum))
        return energy, momentum_norm

    energy0, momentum0 = invariants(omega0)
    energy1, momentum1 = invariants(
        result.samples[-1]["angular_velocity_rad_s"]
    )

    energy_drift = abs(energy1 - energy0) / energy0
    momentum_drift = abs(momentum1 - momentum0) / momentum0

    assert energy_drift < 1e-4, energy_drift
    assert momentum_drift < 1e-4, momentum_drift

    return {
        "relative_rotational_energy_drift": energy_drift,
        "relative_body_angular_momentum_norm_drift": momentum_drift,
        "duration_s": 2.0,
        "dt_s": 0.001,
    }


def check_quaternion_normalisation():
    model = model_with(AerodynamicCoefficients())
    result = model.run(
        make_state(velocity=(5.0, 0.0, 0.0)),
        duration_s=1.0,
        dt_s=0.01,
    )

    errors = [
        abs(float(np.linalg.norm(s["quaternion"])) - 1.0)
        for s in result.samples
    ]
    maximum_error = max(errors)

    assert maximum_error < 1e-10, maximum_error
    return {"maximum_norm_error": maximum_error}


def check_reproducibility():
    model = model_with(AerodynamicCoefficients(drag=0.3))
    initial = make_state(velocity=(12.0, 0.0, 0.0))

    first = model.run(initial, duration_s=1.0, dt_s=0.01)
    second = model.run(initial, duration_s=1.0, dt_s=0.01)

    assert len(first.samples) == len(second.samples)

    for a, b in zip(first.samples, second.samples):
        for key in (
            "position_m",
            "velocity_m_s",
            "quaternion",
            "angular_velocity_rad_s",
            "aerodynamic_force_body_n",
            "aerodynamic_moment_body_nm",
        ):
            assert np.array_equal(a[key], b[key]), key

        for key in (
            "time_s",
            "altitude_m",
            "speed_m_s",
            "airspeed_m_s",
            "mach",
            "dynamic_pressure_pa",
        ):
            assert a[key] == b[key], key

    return {
        "samples": len(first.samples),
        "bitwise_identical_outputs": True,
    }


def main():
    checks = [
        ("analytical_free_fall", check_free_fall),
        ("coordinate_frame_rotation", check_frame_rotation),
        ("timestep_convergence", check_timestep_convergence),
        ("torque_free_rotation", check_torque_free_rotation),
        ("quaternion_normalisation", check_quaternion_normalisation),
        ("deterministic_reproducibility", check_reproducibility),
    ]

    report = {
        "project": "PROJECT 0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Numerical and software verification",
        "validation_status": (
            "Not validated against experimental vehicle flight data."
        ),
        "checks": [],
    }

    failures = 0

    for name, function in checks:
        try:
            details = function()
            record = {
                "name": name,
                "status": "PASS",
                "details": details,
            }
            print(f"PASS: {name}")
        except Exception as exc:
            failures += 1
            record = {
                "name": name,
                "status": "FAIL",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"FAIL: {name}: {exc}")

        report["checks"].append(record)

    report["summary"] = {
        "total": len(checks),
        "passed": len(checks) - failures,
        "failed": failures,
        "status": "PASS" if failures == 0 else "FAIL",
    }

    output = ROOT / "reports" / "verification"
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "verification_report.json"
    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# PROJECT 0 Verification Report",
        "",
        f"- Generated (UTC): {report['created_utc']}",
        f"- Status: **{report['summary']['status']}**",
        f"- Checks passed: {report['summary']['passed']}/{report['summary']['total']}",
        "",
        "## Scope and limitations",
        "",
        "This report covers software and numerical verification under the",
        "specific test conditions described below. It does not validate",
        "the model against experimental flight data or certify a vehicle.",
        "",
    ]

    for item in report["checks"]:
        lines.append(f"## {item['name']}: {item['status']}")
        lines.append("")
        if item["status"] == "PASS":
            lines.append("```json")
            lines.append(json.dumps(item["details"], indent=2))
            lines.append("```")
        else:
            lines.append(f"Error: {item.get('error', 'Unknown error')}")
            lines.append("")
            lines.append("```text")
            lines.append(item.get("traceback", ""))
            lines.append("```")
        lines.append("")

    (output / "verification_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Verification summary: {report['summary']['passed']}/"
        f"{report['summary']['total']} passed"
    )
    print(f"Markdown report: {output / 'verification_report.md'}")
    print(f"JSON report:     {json_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
