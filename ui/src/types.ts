export type Vec3 = [number, number, number];
export type Ring3 = Vec3[];

export type BuildingSurface = {
  category: string;
  semantic_type: string;
  semantic_key: string;
  rings: Ring3[];
};

export type BuildingPart = {
  building_id: string;
  parent_id: string;
  surfaces: BuildingSurface[];
};

export type BuildingPayload = {
  parts: BuildingPart[];
};

export type Panel = {
  name: string;
  col: number;
  row: number;
  width: number;
  height: number;
  area: number;
  is_unique: boolean;
  is_residual_width: boolean;
  is_residual_height: boolean;
  is_specialized: boolean;
  n_vertices: number;
  n_pieces: number;
  polygons_xyz: Ring3[][];
};

export type Wall = {
  wall_id: number;
  wall_type: string;
  n_openings: number;
  n_panels: number;
  n_specialized_panels: number;
  panels: Panel[];
};

export type PanelPart = {
  building_id: string;
  parent_id: string;
  total_panels: number;
  total_unique_panels: number;
  total_specialized_panels: number;
  total_unique_types: number;
  walls: Wall[];
};

export type PanelizationPayload = {
  building_id: string;
  summary: {
    n_parts: number;
    n_walls: number;
    total_panels: number;
    total_unique_panels: number;
    total_specialized_panels: number;
    total_unique_types: number;
    cost_total?: number;
    cost_unique_panel_types?: number;
    cost_panel_elements?: number;
  };
  parts: PanelPart[];
};

export type PanelizeResponse = {
  building?: BuildingPayload;
  panelization: PanelizationPayload;
};

export type Settings = {
  input_json: string;
  selected_building_indices: number[];
  panel_width: number;
  panel_height: number;
  cost_per_unique_panel_type: number;
  cost_per_panel_element: number;
};

export type Layers = {
  roof: boolean;
  wall: boolean;
  reveal: boolean;
  balcony: boolean;
  other: boolean;
  panels: boolean;
  specialized: boolean;
};
