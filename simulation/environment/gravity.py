import numpy as np


EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY_M_S2 = 9.80665


def gravity_acceleration(altitude_m: float):
    altitude = max(0.0, float(altitude_m))

    magnitude = STANDARD_GRAVITY_M_S2 * (
        EARTH_RADIUS_M / (EARTH_RADIUS_M + altitude)
    ) ** 2

    return np.array([0.0, 0.0, -magnitude], dtype=float)
