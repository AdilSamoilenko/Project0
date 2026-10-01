from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MISSION_FILE = ROOT / "requirements" / "mission.yaml"


def load_mission():
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    mission = load_mission()

    print("=" * 64)
    print("PROJECT 0")
    print("EXPERIMENTAL AEROSPACE ENGINEERING PROGRAMME")
    print("=" * 64)
    print()
    print(f"Programme : {mission['project']['name']}")
    print(f"Version   : {mission['project']['version']}")
    print(f"Status    : {mission['project']['status']}")
    print(f"Vehicle   : {mission['vehicle']['designation']}")
    print()
    print("Mission objective:")
    print(f"  {mission['mission']['objective']}")
    print()
    print("Record target:")
    print(f"  {mission['mission']['record']}")
    print()
    print("Engineering pipeline initialised.")


if __name__ == "__main__":
    main()
