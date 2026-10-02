from dataclasses import dataclass


@dataclass(frozen=True)
class AerodynamicCoefficients:
    drag: float = 0.0
    lift: float = 0.0
    side: float = 0.0
    roll_moment: float = 0.0
    pitch_moment: float = 0.0
    yaw_moment: float = 0.0

    def validate(self):
        values = (
            self.drag,
            self.lift,
            self.side,
            self.roll_moment,
            self.pitch_moment,
            self.yaw_moment,
        )

        for value in values:
            if not isinstance(value, (int, float)):
                raise TypeError("Aerodynamic coefficients must be numeric.")
