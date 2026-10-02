from typing import Iterable


def peak_altitude(samples: Iterable) -> float:
    return max(
        float(sample.altitude)
        for sample in samples
    )


def peak_velocity(samples: Iterable) -> float:
    return max(
        float(sample.speed)
        for sample in samples
    )


def flight_duration(samples: Iterable) -> float:
    samples = list(samples)

    if not samples:
        return 0.0

    return float(samples[-1].time - samples[0].time)


def summary(samples: Iterable) -> dict:
    samples = list(samples)

    if not samples:
        return {
            "samples": 0,
            "peak_altitude_m": 0.0,
            "peak_speed_m_s": 0.0,
            "flight_duration_s": 0.0,
        }

    return {
        "samples": len(samples),
        "peak_altitude_m": peak_altitude(samples),
        "peak_speed_m_s": peak_velocity(samples),
        "flight_duration_s": flight_duration(samples),
    }
