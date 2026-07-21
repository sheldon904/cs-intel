import React, { useEffect, useState } from "react";
import { api } from "./api";
import { Loading, fmtNum, fmtPct, StatTile } from "./ui.jsx";

function CausalBars({ causal }) {
  const rows = [
    { label: "Naive (confounded)", v: causal.naive_estimate, c: "#dc2626" },
    { label: "Backdoor-adjusted", v: causal.backdoor_gformula_logistic, c: "#2563eb" },
    { label: "True effect (known)", v: causal.true_ate_risk_diff, c: "#16a34a" },
  ];
  const max = Math.max(...rows.map((r) => Math.abs(r.v))) || 1;
  return (
    <div className="causal-bars">
      {rows.map((r, i) => (
        <div key={i} className="causal-row">
          <div className="causal-label">{r.label}</div>
          <div className="causal-track">
            <div
              className="causal-fill"
              style={{ width: `${(Math.abs(r.v) / max) * 100}%`, background: r.c }}
            />
          </div>
          <div className="causal-val">{(r.v * 100).toFixed(1)}pp</div>
        </div>
      ))}
    </div>
  );
}

export default function Methods() {
  const [churn, setChurn] = useState(null);
  const [intents, setIntents] = useState(null);
  const [causal, setCausal] = useState(null);
  const [llmops, setLlmops] = useState(null);

  useEffect(() => {
    api.churnMetrics().then(setChurn);
    api.intentMetrics().then(setIntents);
    api.causalMetrics().then(setCausal);
    api.llmopsMetrics().then(setLlmops);
  }, []);

  if (!churn || !intents || !causal || !llmops) return <Loading label="Loading metrics…" />;

  const xgb = churn.xgboost || {};
  const log = churn.logistic || {};
  const lift = churn.intent_lift || {};
  const ie = intents.intent || {};
  const se = intents.sentiment || {};

  return (
    <div className="methods">
      {/* Churn model */}
      <div className="card">
        <div className="card-title">Churn model — temporal holdout</div>
        <div className="tiles">
          <StatTile label="XGBoost AUC" value={fmtNum(xgb.auc)} accent="#2563eb" />
          <StatTile label="PR-AUC" value={fmtNum(xgb.pr_auc)} />
          <StatTile label="Brier" value={fmtNum(xgb.brier)} sub="lower = better calibrated" />
          <StatTile label="Logistic AUC" value={fmtNum(log.auc)} sub="baseline" />
          <StatTile
            label="Intent lift"
            value={`+${fmtNum(lift.auc_lift)}`}
            sub="AUC from LLM intents (CV)"
            accent="#16a34a"
          />
        </div>
        <div className="muted sm">
          Strict temporal split: {churn.temporal_split?.train_n} train / {churn.temporal_split?.test_n}{" "}
          test, {churn.temporal_split?.method}. XGBoost is deployed for its better calibration; the
          logistic baseline is competitive on this near-linear signal.
        </div>
        <div className="plots">
          <figure>
            <img src={api.plotUrl("calibration")} alt="calibration curve" />
            <figcaption>Calibration</figcaption>
          </figure>
          <figure>
            <img src={api.plotUrl("pr")} alt="precision-recall" />
            <figcaption>Precision-Recall</figcaption>
          </figure>
          <figure>
            <img src={api.plotUrl("shap")} alt="SHAP summary" />
            <figcaption>SHAP importance</figcaption>
          </figure>
        </div>
      </div>

      {/* Intent eval */}
      <div className="card">
        <div className="card-title">Intent & sentiment labeling — frozen 100-ticket gold set</div>
        <div className="tiles">
          <StatTile label="Intent accuracy" value={fmtPct(ie.accuracy, 1)} />
          <StatTile label="Intent Cohen's κ" value={fmtNum(ie.cohen_kappa, 2)} accent="#2563eb" />
          <StatTile label="Intent macro-F1" value={fmtNum(ie.macro_f1, 2)} />
          <StatTile label="Sentiment accuracy" value={fmtPct(se.accuracy, 1)} />
          <StatTile label="Sentiment κ" value={fmtNum(se.cohen_kappa, 2)} />
        </div>
        <div className="muted sm">
          {intents.note} Feeding these intent rates back into the churn model adds +
          {fmtNum(lift.auc_lift)} AUC.
        </div>
      </div>

      {/* Causal */}
      <div className="card causal-card">
        <div className="card-title">When your churn model lies to you — causal layer</div>
        <p className="muted">
          Naively, high adopters churn {Math.abs(causal.naive_estimate * 100).toFixed(0)} points less
          — but account size confounds it (small accounts both adopt less <em>and</em> churn more).
          A DoWhy backdoor adjustment on account size recovers the true effect:
        </p>
        <CausalBars causal={causal} />
        <div className="muted sm">{causal.interpretation}</div>
        <div className="refutations">
          {Object.entries(causal.refutations || {}).map(([name, r]) => (
            <span key={name} className="refute">
              {name}:{" "}
              {r.error
                ? "n/a"
                : `${(r.estimated_effect * 100).toFixed(1)} → ${(r.new_effect * 100).toFixed(1)}pp`}
            </span>
          ))}
        </div>
      </div>

      {/* LLMOps */}
      <div className="card">
        <div className="card-title">LLMOps — traced on every LLM call</div>
        <div className="tiles">
          <StatTile label="Provider" value={llmops.provider} />
          <StatTile label="Total LLM calls" value={llmops.total_calls?.toLocaleString()} />
          <StatTile
            label="Cost / account"
            value={`$${fmtNum(llmops.projected_cost_per_account, 4)}`}
            sub={llmops.provider === "mock" ? "projected" : "actual"}
          />
          <StatTile label="Accounts scored" value={llmops.n_accounts} />
        </div>
        <table className="stage-table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Calls</th>
              <th>In tok</th>
              <th>Out tok</th>
              <th>Proj cost</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(llmops.by_stage || {}).map(([stage, s]) => (
              <tr key={stage}>
                <td>{stage}</td>
                <td>{s.calls.toLocaleString()}</td>
                <td>{s.input_tokens.toLocaleString()}</td>
                <td>{s.output_tokens.toLocaleString()}</td>
                <td>${s.projected_cost_usd.toFixed(5)}</td>
                <td>{s.avg_latency_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="muted sm">
          {llmops.note} With Langfuse keys set, the same traced calls appear in the Langfuse
          dashboard.
        </div>
      </div>
    </div>
  );
}
