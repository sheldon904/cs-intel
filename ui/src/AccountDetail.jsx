import React, { useEffect, useState } from "react";
import { api } from "./api";
import {
  Gauge,
  ShapBars,
  Sparkline,
  IntentBadge,
  SentimentBadge,
  RiskBadge,
  Loading,
  fmtMoney,
  fmtPct,
} from "./ui.jsx";

const ACTION_LABELS = {
  email_draft: "Send outreach email",
  escalation: "Escalate to save team",
  csm_task: "Create CSM task",
  expansion_outreach: "Expansion outreach",
};

function AgentPanel({ account, onLearned }) {
  const [proposal, setProposal] = useState(account.latest_proposal);
  const [memory, setMemory] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setProposal(account.latest_proposal);
    setMemory(null);
    setNote("");
  }, [account.account_id]);

  const propose = async () => {
    setBusy(true);
    try {
      const res = await api.propose(account.account_id);
      setProposal(res.proposal);
      setMemory(res.memory);
    } finally {
      setBusy(false);
    }
  };

  const decide = async (decision) => {
    setBusy(true);
    try {
      const res = await api.decide(proposal.proposal_id, decision, note);
      setProposal(res.proposal);
      setMemory(res.memory);
      onLearned && onLearned();
    } finally {
      setBusy(false);
    }
  };

  const pending = proposal && proposal.decision === "pending";
  const resolved = proposal && proposal.decision !== "pending";

  return (
    <div className="card agent">
      <div className="card-title">
        <span>🤖 Next-Best-Action Agent</span>
        <span className="muted sm">propose → human approval → learn</span>
      </div>

      {!proposal && (
        <div className="agent-empty">
          <p className="muted">No action proposed yet.</p>
          <button className="btn primary" onClick={propose} disabled={busy}>
            {busy ? "Thinking…" : "Propose next-best-action"}
          </button>
        </div>
      )}

      {proposal && (
        <div className="proposal">
          <div className="proposal-head">
            <span className={`badge action ${proposal.action_type}`}>
              {ACTION_LABELS[proposal.action_type] || proposal.action_type}
            </span>
            <span className={`prio ${proposal.priority}`}>{proposal.priority} priority</span>
          </div>
          <h4>{proposal.title}</h4>
          <pre className="proposal-body">{proposal.body}</pre>
          <div className="rationale">
            <strong>Why:</strong> {proposal.rationale}
          </div>
          {proposal.evidence?.length > 0 && (
            <ul className="evidence">
              {proposal.evidence.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
          {proposal.memory_influence && (
            <div className="memory-influence">🧠 {proposal.memory_influence}</div>
          )}

          {pending && (
            <div className="approval-gate">
              <input
                className="note-input"
                placeholder="Optional note (why approve/reject?)"
                value={note}
                onChange={(e) => setNote(e.target.value)}
              />
              <div className="gate-buttons">
                <button className="btn approve" onClick={() => decide("approved")} disabled={busy}>
                  ✓ Approve
                </button>
                <button className="btn reject" onClick={() => decide("rejected")} disabled={busy}>
                  ✕ Reject
                </button>
                <button className="btn ghost" onClick={propose} disabled={busy}>
                  ↻ Re-propose
                </button>
              </div>
            </div>
          )}

          {resolved && (
            <div className={`decision-banner ${proposal.decision}`}>
              Operator <strong>{proposal.decision}</strong> this action.
              {proposal.operator_note ? ` "${proposal.operator_note}"` : ""}
              <button className="btn ghost sm" onClick={propose} disabled={busy}>
                Propose again
              </button>
            </div>
          )}
        </div>
      )}

      {memory && (
        <div className="memory-panel">
          <div className="memory-title">
            Memory updated · {memory.policy.total_decisions} decision(s) learned
          </div>
          <div className="policy-grid">
            {Object.entries(memory.policy.by_action).map(([action, stat]) => (
              <div key={action} className="policy-row">
                <span className="policy-action">{ACTION_LABELS[action] || action}</span>
                <span className="policy-bar-track">
                  <span
                    className="policy-bar"
                    style={{ width: `${stat.approval_rate * 100}%` }}
                  />
                </span>
                <span className="policy-rate">{fmtPct(stat.approval_rate)}</span>
                <span className="muted sm">
                  {stat.approved}✓ / {stat.rejected}✕
                </span>
              </div>
            ))}
          </div>
          <div className="muted sm">
            Approval rates steer which action the agent proposes next — the system learns from
            every decision.
          </div>
        </div>
      )}
    </div>
  );
}

export default function AccountDetail({ accountId, onLearned }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api.account(accountId).then((d) => {
      if (alive) {
        setDetail(d);
        setLoading(false);
      }
    });
    return () => {
      alive = false;
    };
  }, [accountId]);

  if (loading || !detail) return <Loading label="Loading account…" />;

  const intents = detail.intent_summary || {};
  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <h2>{detail.company_name}</h2>
          <div className="muted">
            {detail.account_id} · {detail.segment} · {detail.industry} · {detail.seats_contracted}{" "}
            seats · {fmtMoney(detail.mrr)}/mo
          </div>
          <div className="muted sm">Renewal: {detail.renewal_date}</div>
        </div>
        <RiskBadge tier={detail.risk_tier} />
      </div>

      <div className="detail-grid">
        <div className="card">
          <div className="card-title">Churn risk</div>
          <div className="gauge-wrap">
            <Gauge value={detail.churn_probability} tier={detail.risk_tier} />
          </div>
        </div>

        <div className="card wide">
          <div className="card-title">Why (SHAP drivers)</div>
          <ShapBars factors={detail.top_factors} />
        </div>

        <div className="card">
          <div className="card-title">Feature adoption trend</div>
          <Sparkline series={detail.usage_series} valueKey="adoption_rate" color="#2563eb" yMax={1} />
          <div className="card-title" style={{ marginTop: 8 }}>
            Support tickets / month
          </div>
          <Sparkline
            series={detail.usage_series}
            valueKey="tickets_opened"
            color="#d97706"
            yMax={Math.max(3, ...detail.usage_series.map((d) => d.tickets_opened || 0))}
          />
        </div>

        <div className="card wide">
          <div className="card-title">
            Intent-tagged tickets{" "}
            <span className="muted sm">
              (cancel-risk {fmtPct(intents.cancel_risk_rate)} · negative{" "}
              {fmtPct(intents.intent_negative_rate)})
            </span>
          </div>
          <div className="tickets">
            {detail.recent_tickets.length === 0 && <div className="muted">No tickets.</div>}
            {detail.recent_tickets.map((t) => (
              <div key={t.ticket_id} className="ticket">
                <div className="ticket-badges">
                  <IntentBadge intent={t.intent} />
                  <SentimentBadge sentiment={t.sentiment} />
                  <span className="muted sm">month {t.month}</span>
                </div>
                <div className="ticket-text">{t.text}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <AgentPanel account={detail} onLearned={onLearned} />
    </div>
  );
}
