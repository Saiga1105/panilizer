import { RefreshCw, Save } from "lucide-react";
import type { Layers, Settings } from "../types";

type ControlsProps = {
  settings: Settings;
  layers: Layers;
  loading: boolean;
  onSettingsChange: (settings: Settings) => void;
  onLayersChange: (layers: Layers) => void;
  onPanelize: () => void;
};

export function Controls({
  settings,
  layers,
  loading,
  onSettingsChange,
  onLayersChange,
  onPanelize,
}: ControlsProps) {
  return (
    <aside className="controls">
      <div className="brand">
        <span>Panilizer</span>
      </div>

      <label className="field">
        <span>Input JSON</span>
        <input
          value={settings.input_json}
          onChange={(event) => onSettingsChange({ ...settings, input_json: event.target.value })}
        />
      </label>

      <div className="sliderGroup">
        <Slider
          label="Panel width"
          value={settings.panel_width}
          min={0.4}
          max={3.0}
          step={0.1}
          unit="m"
          onChange={(panel_width) => onSettingsChange({ ...settings, panel_width })}
        />
        <Slider
          label="Panel height"
          value={settings.panel_height}
          min={0.6}
          max={4.0}
          step={0.1}
          unit="m"
          onChange={(panel_height) => onSettingsChange({ ...settings, panel_height })}
        />
        <Slider
          label="Cost per type"
          value={settings.cost_per_unique_panel_type}
          min={0}
          max={1000}
          step={25}
          unit="EUR"
          decimals={0}
          onChange={(cost_per_unique_panel_type) => onSettingsChange({ ...settings, cost_per_unique_panel_type })}
        />
        <Slider
          label="Cost per element"
          value={settings.cost_per_panel_element}
          min={0}
          max={250}
          step={5}
          unit="EUR"
          decimals={0}
          onChange={(cost_per_panel_element) => onSettingsChange({ ...settings, cost_per_panel_element })}
        />
      </div>

      <button className="primaryButton" onClick={onPanelize} disabled={loading}>
        <RefreshCw size={18} />
        <span>{loading ? "Computing" : "Compute Panels"}</span>
      </button>

      <div className="layers">
        <h2>Layers</h2>
        {Object.entries(layers).map(([key, value]) => (
          <label key={key} className="toggle">
            <span>{layerLabel(key)}</span>
            <input
              type="checkbox"
              checked={value}
              onChange={(event) => onLayersChange({ ...layers, [key]: event.target.checked })}
            />
          </label>
        ))}
      </div>

      <button className="secondaryButton" type="button" disabled>
        <Save size={18} />
        <span>Export via API</span>
      </button>
    </aside>
  );
}

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  decimals?: number;
  onChange: (value: number) => void;
};

function Slider({ label, value, min, max, step, unit, decimals = 1, onChange }: SliderProps) {
  return (
    <label className="slider">
      <span>
        {label}
        <strong>
          {unit === "EUR" ? "EUR " : ""}
          {value.toFixed(decimals)}
          {unit !== "EUR" ? ` ${unit}` : ""}
        </strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function layerLabel(key: string) {
  return key
    .replace("specialized", "specialized panels")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}
