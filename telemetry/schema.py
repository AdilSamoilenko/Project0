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
