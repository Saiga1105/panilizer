import argparse

from src.panelizer import load_panelizer_config, visualize_panels
from src.service import run_panelization


def main():
    parser = argparse.ArgumentParser(description="Create wall panels from LoD3 CityJSON buildings.")
    parser.add_argument(
        "--config",
        default="config/panelizer_config.json",
        help="Path to the panelizer config JSON.",
    )
    args = parser.parse_args()

    config = load_panelizer_config(args.config)
    _, panelization = run_panelization(config, output_json_path=config["output_json"])

    summary = panelization["summary"]
    print(f"Input: {config['input_json']}")
    print(f"Output: {config['output_json']}")
    print(f"Parts: {summary['n_parts']}")
    print(f"Walls: {summary['n_walls']}")
    print(f"Panels: {summary['total_panels']}")
    print(f"Unique panels: {summary['total_unique_panels']}")
    print(f"Unique panel types: {summary['total_unique_types']}")
    print(f"Estimated cost: EUR {summary['cost_total']:,.0f}")

    if config.get("visualize", False):
        visualize_panels(panelization)


if __name__ == "__main__":
    main()
