from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    """
    State vector for the PROJECT 0 flight dynamics simulation.

    Position and velocity are expressed in an inertial Cartesian frame.
    Orientation is represented by a quaternion.

    SI units are used throughout.
    """

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    time: float = 0.0

    @classmethod
    def zero(cls):
        return cls(
            position=np.zeros(3, dtype=float),
            velocity=np.zeros(3, dtype=float),
            quaternion=np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=float,
            ),
            angular_velocity=np.zeros(3, dtype=float),
            time=0.0,
        )

    def copy(self):
        return VehicleState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
        )
