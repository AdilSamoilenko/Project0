from pathlib import Path

import matplotlib.pyplot as plt


def plot_altitude(samples, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = [sample.time for sample in samples]
    altitude = [sample.altitude for sample in samples]

    plt.figure()
    plt.plot(time, altitude)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title("PROJECT 0 Flight Test: Altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_velocity(samples, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = [sample.time for sample in samples]
    speed = [sample.speed for sample in samples]

    plt.figure()
    plt.plot(time, speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("PROJECT 0 Flight Test: Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
