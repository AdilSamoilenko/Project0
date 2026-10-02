from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    quaternion: np.ndarray
    angular_velocity_rad_s: np.ndarray

    def copy(self) -> "VehicleState":
        return VehicleState(
            time_s=float(self.time_s),
            position_m=self.position_m.copy(),
            velocity_m_s=self.velocity_m_s.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity_rad_s=self.angular_velocity_rad_s.copy(),
        )

    def validate(self) -> None:
        arrays = (
            self.position_m,
            self.velocity_m_s,
            self.quaternion,
            self.angular_velocity_rad_s,
        )

        for value in arrays:
            if not np.all(np.isfinite(value)):
                raise ValueError("Vehicle state contains non-finite values.")

        if self.position_m.shape != (3,):
            raise ValueError("Position must contain three components.")

        if self.velocity_m_s.shape != (3,):
            raise ValueError("Velocity must contain three components.")

        if self.quaternion.shape != (4,):
            raise ValueError("Quaternion must contain four components.")

        if self.angular_velocity_rad_s.shape != (3,):
            raise ValueError("Angular velocity must contain three components.")

        norm = np.linalg.norm(self.quaternion)

        if norm < 1e-12:
            raise ValueError("Quaternion magnitude is too small.")
