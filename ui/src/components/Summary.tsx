import type { PanelizationPayload } from "../types";

type SummaryProps = {
  panelization?: PanelizationPayload;
  error?: string;
};

export function Summary({ panelization, error }: SummaryProps) {
  if (error) {
    return <section className="summary error">{error}</section>;
  }

  if (!panelization) {
    return <section className="summary">Waiting for panel data</section>;
  }

  const summary = panelization.summary;
  return (
    <section className="summary">
      <Metric label="Walls" value={summary.n_walls} />
      <Metric label="Panels" value={summary.total_panels} />
      <Metric label="Unique" value={summary.total_unique_panels} />
      <Metric label="Specialized" value={summary.total_specialized_panels} />
      <Metric label="Cost" value={summary.cost_total} prefix="EUR " />
    </section>
  );
}

function Metric({ label, value, prefix = "" }: { label: string; value: number; prefix?: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{prefix}{Math.round(value).toLocaleString()}</strong>
    </div>
  );
}
