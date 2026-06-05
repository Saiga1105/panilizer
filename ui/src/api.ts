import type { PanelizeResponse, Settings } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export async function panelize(settings: Settings): Promise<PanelizeResponse> {
  const response = await fetch(`${API_BASE}/panelize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_json: settings.input_json,
      selected_building_indices: settings.selected_building_indices,
      include_building: true,
      settings: {
        panel_width: settings.panel_width,
        panel_height: settings.panel_height,
        cost_per_unique_panel_type: settings.cost_per_unique_panel_type,
        cost_per_panel_element: settings.cost_per_panel_element,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Panelization failed: ${response.status}`);
  }

  return response.json();
}
