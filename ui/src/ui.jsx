// Small dependency-free UI atoms (badges, gauge, sparklines) shared across views.
import React from "react";

export const fmtPct = (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
export const fmtMoney = (x) =>
  x == null ? "–" : `$${Math.round(x).toLocaleString()}`;
export const fmtNum = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));

const RISK_COLORS = { High: "#dc2626", Medium: "#d97706", Low: "#16a34a" };
const INTENT_COLORS = {
  "cancel-risk": "#dc2626",
  "expansion-signal": "#16a34a",
  confusion: "#2563eb",
  "bug-report": "#7c3aed",
};
const SENT_COLORS = { negative: "#dc2626", neutral: "#64748b", positive: "#16a34a" };

export function RiskBadge({ tier }) {
  return (
    <span className="badge" style={{ background: RISK_COLORS[tier] || "#64748b" }}>
      {tier}
    </span>
  );
}

export function IntentBadge({ intent }) {
  return (
    <span className="badge sm" style={{ background: INTENT_COLORS[intent] || "#64748b" }}>
      {intent}
    </span>
  );
}

export function SentimentBadge({ sentiment }) {
  return (
    <span className="badge sm outline" style={{ color: SENT_COLORS[sentiment], borderColor: SENT_COLORS[sentiment] }}>
      {sentiment}
    </span>
  );
}

export function StatTile({ label, value, sub, accent }) {
  return (
    <div className="tile">
      <div className="tile-value" style={accent ? { color: accent } : undefined}>{value}</div>
      <div className="tile-label">{label}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

// Semicircular risk gauge.
export function Gauge({ value, tier }) {
  const r = 70;
  const cx = 90;
  const cy = 90;
  const angle = Math.PI * (1 - value); // value 0..1 -> pi..0
  const x = cx + r * Math.cos(angle);
  const y = cy - r * Math.sin(angle);
  const color = RISK_COLORS[tier] || "#64748b";
  const arc = (frac, stroke, width) => {
    const a = Math.PI * (1 - frac);
    const ex = cx + r * Math.cos(a);
    const ey = cy - r * Math.sin(a);
    const large = frac > 0.5 ? 1 : 0;
    return (
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeLinecap="round"
      />
    );
  };
  return (
    <svg viewBox="0 0 180 110" width="180" height="110">
      {arc(1, "#e2e8f0", 12)}
      {arc(Math.max(0.001, value), color, 12)}
      <circle cx={x} cy={y} r="6" fill={color} />
      <text x={cx} y={cy - 8} textAnchor="middle" className="gauge-value" fill={color}>
        {fmtPct(value)}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" className="gauge-sub">
        churn risk
      </text>
    </svg>
  );
}

// SHAP-style horizontal contribution bars.
export function ShapBars({ factors }) {
  if (!factors || !factors.length) return null;
  const max = Math.max(...factors.map((f) => Math.abs(f.shap_value))) || 1;
  const label = (f) => f.feature.replace(/_/g, " ");
  return (
    <div className="shap">
      {factors.map((f, i) => {
        const w = (Math.abs(f.shap_value) / max) * 100;
        const up = f.shap_value > 0;
        return (
          <div key={i} className="shap-row">
            <div className="shap-label">{label(f)}</div>
            <div className="shap-track">
              <div
                className="shap-fill"
                style={{ width: `${w}%`, background: up ? "#dc2626" : "#16a34a" }}
              />
            </div>
            <div className="shap-val" style={{ color: up ? "#dc2626" : "#16a34a" }}>
              {up ? "+" : ""}
              {f.shap_value.toFixed(2)}
            </div>
          </div>
        );
      })}
      <div className="shap-legend">
        <span style={{ color: "#dc2626" }}>■ increases risk</span>
        <span style={{ color: "#16a34a" }}>■ decreases risk</span>
      </div>
    </div>
  );
}

// Simple line sparkline for a 0..1 series.
export function Sparkline({ series, valueKey, color = "#2563eb", height = 60, yMax = 1 }) {
  if (!series || series.length < 2) return <div className="muted sm">not enough data</div>;
  const w = 260;
  const h = height;
  const xs = series.map((_, i) => (i / (series.length - 1)) * (w - 8) + 4);
  const ys = series.map((d) => {
    const v = d[valueKey] == null ? 0 : d[valueKey];
    return h - 6 - (Math.min(v, yMax) / yMax) * (h - 12);
  });
  const path = xs.map((x, i) => `${i ? "L" : "M"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="3" fill={color} />
    </svg>
  );
}

export function Loading({ label = "Loading…" }) {
  return <div className="muted center pad">{label}</div>;
}
