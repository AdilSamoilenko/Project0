from dataclasses import dataclass
import math
import numpy as np

from simulation.core.state import VehicleState
from simulation.core.frames import quaternion_to_rotation_matrix
from simulation.environment.environment import EnvironmentModel
from simulation.aerodynamics import (
    AerodynamicCoefficients,
    AerodynamicModel,
    calculate_aerodynamic_forces,
)


@dataclass
class IntegrationResult:
    samples: list
    termination_reason: str


class IntegratedFlightModel:
    """Generic passive 6-DOF model with environment and coefficient aerodynamics.

    This is a software integration model, not a validated vehicle predictor.
    No propulsion model is included.
    """

    def __init__(
        self,
        mass_kg=1.0,
        inertia_kg_m2=(0.1, 0.1, 0.01),
        reference_area_m2=0.01,
        reference_length_m=1.0,
        coefficients=None,
        environment=None,
    ):
        if not math.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError("Mass must be finite and positive.")

        inertia = np.asarray(inertia_kg_m2, dtype=float)
        if inertia.shape != (3,) or not np.all(np.isfinite(inertia)):
            raise ValueError("Inertia must contain three finite values.")
        if np.any(inertia <= 0):
            raise ValueError("Principal inertias must be positive.")

        self.mass_kg = float(mass_kg)
        self.inertia = inertia
        self.environment = environment or EnvironmentModel()
        self.aerodynamics = AerodynamicModel(
            reference_area_m2=reference_area_m2,
            reference_length_m=reference_length_m,
            coefficients=coefficients or AerodynamicCoefficients(
                drag=0.30,
                lift=0.0,
                side=0.0,
                roll_moment=0.0,
                pitch_moment=0.0,
                yaw_moment=0.0,
            ),
        )

    @staticmethod
    def _normalise_quaternion(q):
        norm = float(np.linalg.norm(q))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Invalid attitude quaternion.")
        return q / norm

    @staticmethod
    def _quaternion_derivative(q, omega):
        w, x, y, z = q
        p, r, s = omega
        return 0.5 * np.array([
            -x*p - y*r - z*s,
             w*p + y*s - z*r,
             w*r + z*p - x*s,
             w*s + x*r - y*p,
        ], dtype=float)

    @staticmethod
    def _field(obj, names):
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"Environment state missing expected field: {names}"
        )

    def _derivative(self, time_s, y):
        position = y[0:3]
        velocity = y[3:6]
        q = self._normalise_quaternion(y[6:10])
        omega = y[10:13]

        env = self.environment.evaluate(
            time_s=time_s,
            position_m=position,
            velocity_m_s=velocity,
        )

        wind = np.asarray(self._field(
            env, ("wind_velocity_m_s", "wind_velocity")
        ), dtype=float)
        relative_velocity = velocity - wind
        airspeed = float(np.linalg.norm(relative_velocity))

        density = float(self._field(env, ("density_kg_m3", "density")))
        speed_of_sound = float(self._field(
            env, ("speed_of_sound_m_s", "speed_of_sound")
        ))
        dynamic_pressure = float(self._field(
            env, ("dynamic_pressure_pa",)
        ))
        gravity = np.asarray(self._field(
            env, ("gravity_vector_m_s2", "gravity_m_s2")
        ), dtype=float)

        if gravity.shape == ():
            gravity = np.array([0.0, 0.0, -float(gravity)])

        if gravity.shape != (3,):
            raise ValueError("Gravity must be a three-component vector.")

        if airspeed > 1e-9:
            body_velocity = quaternion_to_rotation_matrix(q).T @ relative_velocity
            alpha = math.atan2(-body_velocity[2], body_velocity[0])
            beta = math.atan2(
                body_velocity[1],
                math.hypot(body_velocity[0], body_velocity[2]),
            )
        else:
            alpha = 0.0
            beta = 0.0

        mach = airspeed / speed_of_sound if speed_of_sound > 0 else 0.0

        aero_state = self.aerodynamics.build_state(
            airspeed_m_s=airspeed,
            mach=mach,
            angle_of_attack_rad=alpha,
            sideslip_rad=beta,
            dynamic_pressure_pa=dynamic_pressure,
        )

        aero = calculate_aerodynamic_forces(
            dynamic_pressure_pa=aero_state.dynamic_pressure_pa,
            reference_area_m2=self.aerodynamics.reference_area_m2,
            reference_length_m=self.aerodynamics.reference_length_m,
            coefficients=self.aerodynamics.coefficients,
        )

        rotation = quaternion_to_rotation_matrix(q)
        force_inertial = rotation @ aero.force_body_n
        acceleration = gravity + force_inertial / self.mass_kg

        angular_momentum = self.inertia * omega
        angular_acceleration = (
            aero.moment_body_nm - np.cross(omega, angular_momentum)
        ) / self.inertia

        derivative = np.concatenate((
            velocity,
            acceleration,
            self._quaternion_derivative(q, omega),
            angular_acceleration,
        ))

        diagnostics = {
            "altitude_m": float(position[2]),
            "speed_m_s": float(np.linalg.norm(velocity)),
            "airspeed_m_s": airspeed,
            "mach": mach,
            "dynamic_pressure_pa": dynamic_pressure,
            "density_kg_m3": density,
            "aerodynamic_force_body_n": aero.force_body_n.copy(),
            "aerodynamic_moment_body_nm": aero.moment_body_nm.copy(),
        }
        return derivative, diagnostics

    def run(self, initial_state, duration_s=5.0, dt_s=0.01):
        if duration_s <= 0 or dt_s <= 0:
            raise ValueError("Duration and timestep must be positive.")
        if not math.isfinite(duration_s) or not math.isfinite(dt_s):
            raise ValueError("Duration and timestep must be finite.")

        state = initial_state.copy()
        state.validate()

        y = np.concatenate((
            np.asarray(state.position_m, dtype=float),
            np.asarray(state.velocity_m_s, dtype=float),
            self._normalise_quaternion(
                np.asarray(state.quaternion, dtype=float)
            ),
            np.asarray(state.angular_velocity_rad_s, dtype=float),
        ))

        samples = []
        time_s = float(state.time_s)
        end_time = time_s + float(duration_s)

        while time_s <= end_time + 1e-10:
            derivative, diagnostics = self._derivative(time_s, y)
            diagnostics["time_s"] = time_s
            diagnostics["position_m"] = y[0:3].copy()
            diagnostics["velocity_m_s"] = y[3:6].copy()
            diagnostics["quaternion"] = y[6:10].copy()
            diagnostics["angular_velocity_rad_s"] = y[10:13].copy()
            samples.append(diagnostics)

            if time_s >= end_time - 1e-10:
                break

            h = min(float(dt_s), end_time - time_s)

            k1 = derivative
            k2, _ = self._derivative(time_s + h/2, y + h*k1/2)
            k3, _ = self._derivative(time_s + h/2, y + h*k2/2)
            k4, _ = self._derivative(time_s + h, y + h*k3)

            y = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
            y[6:10] = self._normalise_quaternion(y[6:10])
            time_s += h

            if not np.all(np.isfinite(y)):
                raise FloatingPointError(
                    "Non-finite state encountered during integration."
                )

        return IntegrationResult(
            samples=samples,
            termination_reason="duration_reached",
        )
