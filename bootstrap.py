from pathlib import Path

ROOT = Path(__file__).resolve().parent

directories = [
    "docs",
    "requirements",
    "vehicle",
    "vehicle/config",
    "vehicle/geometry",
    "vehicle/mass",
    "openrocket",
    "gazebo",
    "gazebo/worlds",
    "gazebo/models",
    "gazebo/plugins",
    "gazebo/launch",
    "simulation",
    "simulation/core",
    "simulation/models",
    "simulation/analysis",
    "simulation/monte_carlo",
    "simulation/output",
    "avionics",
    "avionics/architecture",
    "avionics/firmware",
    "avionics/telemetry",
    "structures",
    "structures/CAD",
    "structures/FEA",
    "recovery",
    "telemetry",
    "telemetry/raw",
    "telemetry/processed",
    "telemetry/analysis",
    "testing",
    "analysis",
    "reports",
    "tools",
    "results",
    "data",
    "data/raw",
    "data/processed",
]

for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)


files = {
    "simulation/__init__.py": "",
    "simulation/core/__init__.py": "",
    "simulation/models/__init__.py": "",
    "simulation/analysis/__init__.py": "",
    "simulation/monte_carlo/__init__.py": "",
    "vehicle/__init__.py": "",
    "telemetry/__init__.py": "",
    "results/.gitkeep": "",
    "data/raw/.gitkeep": "",
}


for filename, content in files.items():
    path = ROOT / filename
    if not path.exists():
        path.write_text(content, encoding="utf-8")


print("PROJECT 0 BOOTSTRAP COMPLETE")
print()
print(f"Root: {ROOT}")
print()
print("Directories created:")
for directory in directories:
    print(f"  [OK] {directory}")
