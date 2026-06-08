from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_models import BuildingRequest, PanelizeRequest
from .panelizer import load_panelizer_config
from .service import (
    building_to_viewer_payload,
    load_buildings,
    panelization_to_viewer_payload,
    run_panelization,
)


app = FastAPI(title="Panilizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    config_data = load_panelizer_config("config/panelizer_config.json")
    return {
        key: value
        for key, value in config_data.items()
        if key not in {"tolerance", "precision"}
    }


@app.post("/buildings")
def buildings(request: BuildingRequest):
    selected = load_buildings(request.input_json, request.selected_building_indices)
    return building_to_viewer_payload(selected)


@app.post("/panelize")
def panelize(request: PanelizeRequest):
    config_data = _request_to_config(request)
    buildings_data, panelization = run_panelization(config_data)
    response = {"panelization": panelization_to_viewer_payload(panelization)}
    if request.include_building:
        response["building"] = building_to_viewer_payload(buildings_data)
    return response


@app.post("/panelize/export")
def panelize_export(request: PanelizeRequest):
    config_data = _request_to_config(request)
    output_json = request.output_json or config_data["output_json"]
    buildings_data, panelization = run_panelization(config_data, output_json_path=output_json)
    return {
        "output_json": str(Path(output_json)),
        "panelization": panelization_to_viewer_payload(panelization),
        "building": building_to_viewer_payload(buildings_data) if request.include_building else None,
    }


def _request_to_config(request: PanelizeRequest) -> dict:
    config_data = load_panelizer_config("config/panelizer_config.json")
    config_data.update({
        "input_json": request.input_json,
        "selected_building_indices": request.selected_building_indices,
        "panel_width": request.settings.panel_width,
        "panel_height": request.settings.panel_height,
        "cost_per_unique_panel_type": request.settings.cost_per_unique_panel_type,
        "cost_per_panel_element": request.settings.cost_per_panel_element,
    })
    if request.output_json is not None:
        config_data["output_json"] = request.output_json
    return config_data
