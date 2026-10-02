import csv
from pathlib import Path

from simulation.core.flight import FlightSample


class TelemetryRecorder:

    @staticmethod
    def samples_to_rows(samples: list[FlightSample]):

        rows = []

        for sample in samples:

            rows.append(
                {
                    "time_s": sample.time,
                    "x_m": sample.position[0],
                    "y_m": sample.position[1],
                    "z_m": sample.position[2],
                    "vx_m_s": sample.velocity[0],
                    "vy_m_s": sample.velocity[1],
                    "vz_m_s": sample.velocity[2],
                    "ax_m_s2": sample.acceleration[0],
                    "ay_m_s2": sample.acceleration[1],
                    "az_m_s2": sample.acceleration[2],
                    "altitude_m": sample.altitude,
                    "speed_m_s": sample.speed,
                    "mach": sample.mach,
                    "dynamic_pressure_pa": sample.dynamic_pressure,
                }
            )

        return rows

    @staticmethod
    def save_csv(samples: list[FlightSample], path: str | Path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = TelemetryRecorder.samples_to_rows(samples)

        if not rows:
            return

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)
