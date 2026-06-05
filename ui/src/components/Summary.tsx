import type { PanelizationPayload, Settings } from "../types";

type SummaryProps = {
  panelization?: PanelizationPayload;
  settings: Settings;
  error?: string;
};

export function Summary({ panelization, settings, error }: SummaryProps) {
  if (error) {
    return <section className="summary error">{error}</section>;
  }

  if (!panelization) {
    return <section className="summary">Waiting for panel data</section>;
  }

  const summary = panelization.summary;
  const fallbackCost =
    summary.total_unique_types * settings.cost_per_unique_panel_type
    + summary.total_panels * settings.cost_per_panel_element;
  const totalCost =
    typeof summary.cost_total === "number" && Number.isFinite(summary.cost_total)
      ? summary.cost_total
      : fallbackCost;

  return (
    <section className="summary">
      <Metric label="Walls" value={summary.n_walls} />
      <Metric label="Panels" value={summary.total_panels} />
      <Metric label="Unique" value={summary.total_unique_panels} />
      <Metric label="Specialized" value={summary.total_specialized_panels} />
      <Metric label="Cost" value={totalCost} prefix="EUR " />
    </section>
  );
}

function Metric({ label, value, prefix = "" }: { label: string; value: number; prefix?: string }) {
  const displayValue = Number.isFinite(value) ? Math.round(value).toLocaleString() : "0";
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{prefix}{displayValue}</strong>
    </div>
  );
}
