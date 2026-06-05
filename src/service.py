from pathlib import Path

from .building_parser import build_lod3_building_dictionaries, load_json_file
from .panelizer import load_panelizer_config, panelize_buildings


def select_buildings(buildings: list[dict], selected_indices: list[int] | None) -> list[dict]:
    if selected_indices is None:
        return buildings
    return [buildings[index] for index in selected_indices]


def load_buildings(input_json: Path | str, selected_indices: list[int] | None = None) -> list[dict]:
    input_json = Path(input_json)
    cityjson = load_json_file(input_json)
    buildings = build_lod3_building_dictionaries({input_json: cityjson})
    return select_buildings(buildings, selected_indices)


def run_panelization(config: dict, output_json_path: Path | str | None = None) -> tuple[list[dict], dict]:
    buildings = load_buildings(config["input_json"], config.get("selected_building_indices"))
    panelization = panelize_buildings(buildings, config=config, output_json_path=output_json_path)
    return buildings, panelization


def load_config_and_run(config_path: Path | str) -> tuple[list[dict], dict]:
    config = load_panelizer_config(config_path)
    return run_panelization(config, output_json_path=config["output_json"])


def building_to_viewer_payload(buildings: list[dict]) -> dict:
    parts = []
    categories = ["roof", "wall", "reveal", "balcony", "other"]
    for building in buildings:
        surfaces = []
        for category in categories:
            for surface in building.get("surfaces", {}).get(category, []):
                surfaces.append({
                    "category": category,
                    "semantic_type": surface["semantic_type"],
                    "semantic_key": surface["semantic_key"],
                    "rings": surface["rings"],
                })

        parts.append({
            "building_id": building["id"],
            "parent_id": str(building["id"]).split("-")[0],
            "surfaces": surfaces,
        })

    return {"parts": parts}


def panelization_to_viewer_payload(panelization: dict) -> dict:
    return {
        "building_id": panelization["building_id"],
        "summary": panelization["summary"],
        "parts": panelization["parts"],
    }
