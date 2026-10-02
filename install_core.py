from pathlib import Path

ROOT = Path(__file__).resolve().parent

files = {

"simulation/core/frames.py": r'''
import numpy as np


def normalise_quaternion(q):
    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,).")

    norm = np.linalg.norm(q)

    if norm == 0:
        raise ValueError("Quaternion cannot have zero magnitude.")

    return q / norm


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quaternion_to_rotation_matrix(q):
    q = normalise_quaternion(q)

    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def body_to_inertial(vector_body, quaternion):
    rotation = quaternion_to_rotation_matrix(quaternion)
    return rotation @ np.asarray(vector_body, dtype=float)


def inertial_to_body(vector_inertial, quaternion):
    rotation = quaternion_to_rotation_matrix(quaternion)
    return rotation.T @ np.asarray(vector_inertial, dtype=float)
''',

"simulation/models/gravity.py": r'''
import numpy as np


EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY = 9.80665


def gravity_magnitude(altitude_m):
    """
    Approximate gravitational acceleration as a function of altitude.
    """

    radius = EARTH_RADIUS_M + max(float(altitude_m), 0.0)

    return STANDARD_GRAVITY * (
        EARTH_RADIUS_M / radius
    ) ** 2


def gravity_vector(altitude_m):
    return np.array([
        0.0,
        0.0,
        -gravity_magnitude(altitude_m),
    ])
''',

"simulation/models/aerodynamics.py": r'''
from dataclasses import dataclass
import numpy as np

from simulation.models.atmosphere import standard_atmosphere


@dataclass
class AeroState:
    altitude_m: float
    speed_m_s: float
    mach: float
    dynamic_pressure_pa: float
    density_kg_m3: float


def calculate_aero_state(
    altitude_m,
    velocity_vector,
):
    velocity_vector = np.asarray(
        velocity_vector,
        dtype=float,
    )

    speed = float(np.linalg.norm(velocity_vector))

    atmosphere = standard_atmosphere(altitude_m)

    if atmosphere.speed_of_sound_m_s > 0:
        mach = speed / atmosphere.speed_of_sound_m_s
    else:
        mach = 0.0

    dynamic_pressure = (
        0.5
        * atmosphere.density_kg_m3
        * speed ** 2
    )

    return AeroState(
        altitude_m=float(altitude_m),
        speed_m_s=speed,
        mach=mach,
        dynamic_pressure_pa=dynamic_pressure,
        density_kg_m3=atmosphere.density_kg_m3,
    )
''',

"simulation/core/integrator.py": r'''
import numpy as np


class RK4:
    """
    Generic fourth-order Runge-Kutta integrator.

    The derivative function must accept:
        f(t, y) -> dy/dt
    """

    @staticmethod
    def step(function, t, y, dt):
        y = np.asarray(y, dtype=float)

        k1 = np.asarray(
            function(t, y),
            dtype=float,
        )

        k2 = np.asarray(
            function(t + dt/2, y + dt*k1/2),
            dtype=float,
        )

        k3 = np.asarray(
            function(t + dt/2, y + dt*k2/2),
            dtype=float,
        )

        k4 = np.asarray(
            function(t + dt, y + dt*k3),
            dtype=float,
        )

        return y + (
            dt / 6.0
        ) * (
            k1 + 2*k2 + 2*k3 + k4
        )
''',

"telemetry/schema.py": r'''
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class TelemetryRecord:
    time_s: float
    altitude_m: float
    velocity_m_s: float

    acceleration_m_s2: Optional[float] = None
    pressure_pa: Optional[float] = None
    temperature_k: Optional[float] = None
    mach: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
        )
''',

"simulation/analysis/metrics.py": r'''
import numpy as np


def peak_altitude(records):
    if not records:
        return None

    return max(
        record["altitude_m"]
        for record in records
    )


def peak_velocity(records):
    if not records:
        return None

    return max(
        record["velocity_m_s"]
        for record in records
    )


def flight_duration(records):
    if not records:
        return 0.0

    return (
        records[-1]["time_s"]
        - records[0]["time_s"]
    )


def summary(records):
    return {
        "samples": len(records),
        "duration_s": flight_duration(records),
        "peak_altitude_m": peak_altitude(records),
        "peak_velocity_m_s": peak_velocity(records),
    }
''',

"tools/test_core.py": r'''
import numpy as np

from simulation.core.frames import (
    normalise_quaternion,
    quaternion_to_rotation_matrix,
)

from simulation.models.atmosphere import (
    standard_atmosphere,
)

from simulation.models.aerodynamics import (
    calculate_aero_state,
)

from simulation.models.gravity import (
    gravity_magnitude,
)


def main():

    q = normalise_quaternion(
        np.array([2.0, 0.0, 0.0, 0.0])
    )

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    rotation = quaternion_to_rotation_matrix(q)

    assert np.allclose(
        rotation,
        np.eye(3),
    )

    atmosphere = standard_atmosphere(0.0)

    assert atmosphere.pressure_pa > 100000.0
    assert atmosphere.density_kg_m3 > 1.0

    aero = calculate_aero_state(
        0.0,
        np.array([100.0, 0.0, 0.0]),
    )

    assert aero.speed_m_s == 100.0
    assert aero.mach > 0.0
    assert aero.dynamic_pressure_pa > 0.0

    assert 9.0 < gravity_magnitude(0.0) < 10.0

    print("PROJECT 0 CORE TESTS")
    print("====================")
    print("[PASS] Quaternion mathematics")
    print("[PASS] Coordinate-frame mathematics")
    print("[PASS] Standard atmosphere")
    print("[PASS] Aerodynamic state")
    print("[PASS] Gravity model")
    print()
    print("ALL CORE TESTS PASSED")


if __name__ == "__main__":
    main()
''',

}

for filename, content in files.items():

    path = ROOT / filename
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8",
    )

print("PROJECT 0 SIMULATION CORE INSTALLED")
print(f"Files installed: {len(files)}")
