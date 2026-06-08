import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.panelizer import load_panelizer_config
from src.service import building_to_viewer_payload, panelization_to_viewer_payload, run_panelization


def main():
    config = load_panelizer_config("config/panelizer_config.json")
    buildings, panelization = run_panelization(config)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    with open(docs_dir / "demo_building.json", "w", encoding="utf-8") as handle:
        json.dump(building_to_viewer_payload(buildings), handle, indent=2)

    with open(docs_dir / "demo_panels.json", "w", encoding="utf-8") as handle:
        json.dump(panelization_to_viewer_payload(panelization), handle, indent=2)

    print("Exported docs/demo_building.json and docs/demo_panels.json")


if __name__ == "__main__":
    main()
