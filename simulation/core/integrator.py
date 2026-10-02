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
