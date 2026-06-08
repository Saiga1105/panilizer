from pydantic import BaseModel, Field


class PanelizerSettings(BaseModel):
    panel_width: float = Field(default=1.2, gt=0)
    panel_height: float = Field(default=2.4, gt=0)
    cost_per_unique_panel_type: float = Field(default=250.0, ge=0)
    cost_per_panel_element: float = Field(default=45.0, ge=0)


class PanelizeRequest(BaseModel):
    input_json: str = "input/outputID2/Output/lod3.json"
    output_json: str | None = None
    selected_building_indices: list[int] | None = [0]
    settings: PanelizerSettings = Field(default_factory=PanelizerSettings)
    include_building: bool = True


class BuildingRequest(BaseModel):
    input_json: str = "input/outputID2/Output/lod3.json"
    selected_building_indices: list[int] | None = [0]
