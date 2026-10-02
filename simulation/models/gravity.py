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
