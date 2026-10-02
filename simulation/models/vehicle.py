from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.core.dynamics import RigidBodyParameters


@dataclass
class VehicleModel:
    mass_kg: float
    inertia_xx_kg_m2: float
    inertia_yy_kg_m2: float
    inertia_zz_kg_m2: float
    reference_area_m2: float

    def rigid_body_parameters(self) -> RigidBodyParameters:
        inertia = np.diag([
            self.inertia_xx_kg_m2,
            self.inertia_yy_kg_m2,
            self.inertia_zz_kg_m2,
        ])

        return RigidBodyParameters(
            mass_kg=self.mass_kg,
            inertia_kg_m2=inertia,
        )

    def validate(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("Vehicle mass must be positive.")

        if self.reference_area_m2 <= 0:
            raise ValueError("Reference area must be positive.")

        if min(
            self.inertia_xx_kg_m2,
            self.inertia_yy_kg_m2,
            self.inertia_zz_kg_m2,
        ) <= 0:
            raise ValueError("All principal inertias must be positive.")
