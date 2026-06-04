import argparse
from pathlib import Path

from src.building_parser import build_lod3_building_dictionaries, load_json_file
from src.panelizer import load_panelizer_config, panelize_buildings, visualize_panels


def main():
    parser = argparse.ArgumentParser(description="Create wall panels from LoD3 CityJSON buildings.")
    parser.add_argument(
        "--config",
        default="config/panelizer_config.json",
        help="Path to the panelizer config JSON.",
    )
    args = parser.parse_args()

    config = load_panelizer_config(args.config)
    input_json = Path(config["input_json"])
    output_json = Path(config["output_json"])

    cityjson = load_json_file(input_json)
    buildings = build_lod3_building_dictionaries({input_json: cityjson})
    selected_indices = config.get("selected_building_indices")
    if selected_indices is not None:
        buildings = [buildings[index] for index in selected_indices]

    panelization = panelize_buildings(buildings, config=config, output_json_path=output_json)

    summary = panelization["summary"]
    print(f"Input: {input_json}")
    print(f"Output: {output_json}")
    print(f"Parts: {summary['n_parts']}")
    print(f"Walls: {summary['n_walls']}")
    print(f"Panels: {summary['total_panels']}")
    print(f"Unique panels: {summary['total_unique_panels']}")
    print(f"Unique panel types: {summary['total_unique_types']}")
    print(f"Skipped outside wall polygon: {summary['total_skipped']}")

    if config.get("visualize", False):
        visualize_panels(panelization)


if __name__ == "__main__":
    main()
