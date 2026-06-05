import { useCallback, useEffect, useState } from "react";
import { panelize } from "./api";
import { Controls } from "./components/Controls";
import { Summary } from "./components/Summary";
import { Viewer } from "./components/Viewer";
import type { Layers, PanelizeResponse, Settings } from "./types";

const initialSettings: Settings = {
  input_json: "input/outputID1/Output/lod3.json",
  selected_building_indices: [0],
  panel_width: 1.2,
  panel_height: 2.4,
  cost_per_unique_panel_type: 250,
  cost_per_panel_element: 45,
};

const initialLayers: Layers = {
  roof: true,
  wall: true,
  reveal: true,
  balcony: true,
  other: false,
  panels: true,
  specialized: true,
};

export default function App() {
  const [settings, setSettings] = useState(initialSettings);
  const [layers, setLayers] = useState(initialLayers);
  const [data, setData] = useState<PanelizeResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const compute = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      setData(await panelize(settings));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Panelization failed");
    } finally {
      setLoading(false);
    }
  }, [settings]);

  useEffect(() => {
    void compute();
  }, []);

  return (
    <main className="appShell">
      <Controls
        settings={settings}
        layers={layers}
        loading={loading}
        onSettingsChange={setSettings}
        onLayersChange={setLayers}
        onPanelize={compute}
      />
      <section className="workspace">
        <div className="viewerFrame">
          <Viewer building={data?.building} panelization={data?.panelization} layers={layers} />
        </div>
        <Summary panelization={data?.panelization} error={error} />
      </section>
    </main>
  );
}
