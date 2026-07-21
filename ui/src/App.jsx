import React, { useEffect, useState } from "react";
import { api } from "./api";
import AccountDetail from "./AccountDetail.jsx";
import Methods from "./Methods.jsx";
import { RiskBadge, StatTile, Loading, fmtMoney, fmtPct, fmtNum } from "./ui.jsx";

function Sidebar({ accounts, selected, onSelect, filter, setFilter }) {
  const tiers = ["High", "Medium", "Low"];
  return (
    <aside className="sidebar">
      <div className="filter-row">
        <button className={`chip ${!filter ? "on" : ""}`} onClick={() => setFilter(null)}>
          All
        </button>
        {tiers.map((t) => (
          <button key={t} className={`chip ${filter === t ? "on" : ""}`} onClick={() => setFilter(t)}>
            {t}
          </button>
        ))}
      </div>
      <div className="account-list">
        {accounts.map((a) => (
          <button
            key={a.account_id}
            className={`account-item ${selected === a.account_id ? "sel" : ""}`}
            onClick={() => onSelect(a.account_id)}
          >
            <div className="ai-main">
              <span className="ai-name">{a.company_name}</span>
              <RiskBadge tier={a.risk_tier} />
            </div>
            <div className="ai-sub muted">
              {a.segment} · {fmtMoney(a.mrr)}/mo · {fmtPct(a.churn_probability)} risk
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}

export default function App() {
  const [tab, setTab] = useState("portfolio");
  const [health, setHealth] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [filter, setFilter] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    api.health().then(setHealth);
  }, []);

  useEffect(() => {
    api.portfolio().then(setPortfolio);
  }, []);

  useEffect(() => {
    api.accounts(filter).then((a) => {
      setAccounts(a);
      if (!selected && a.length) setSelected(a[0].account_id);
    });
  }, [filter]);

  const refreshPortfolio = () => api.portfolio().then(setPortfolio);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◆</span>
          <span>cs-intel</span>
          <span className="tagline">Customer Success Intelligence</span>
        </div>
        <nav className="tabs">
          <button className={tab === "portfolio" ? "on" : ""} onClick={() => setTab("portfolio")}>
            Portfolio
          </button>
          <button className={tab === "methods" ? "on" : ""} onClick={() => setTab("methods")}>
            Model &amp; Methods
          </button>
        </nav>
        <div className="provider">
          {health && (
            <>
              <span className={`dot ${health.langfuse_enabled ? "green" : "amber"}`} />
              LLM: {health.llm_provider}
              {health.langfuse_enabled ? " · Langfuse on" : ""}
            </>
          )}
        </div>
      </header>

      {tab === "portfolio" && (
        <>
          <section className="summary-bar">
            {portfolio ? (
              <div className="tiles">
                <StatTile label="Accounts" value={portfolio.summary.n_accounts} />
                <StatTile
                  label="High risk"
                  value={portfolio.summary.n_high_risk}
                  accent="#dc2626"
                />
                <StatTile
                  label="MRR at risk"
                  value={fmtMoney(portfolio.summary.mrr_at_risk)}
                  sub="high-risk accounts"
                />
                <StatTile
                  label="Avg churn risk"
                  value={fmtPct(portfolio.summary.avg_churn_probability)}
                />
                <StatTile label="Churn AUC" value={fmtNum(portfolio.churn_auc, 3)} accent="#2563eb" />
                <StatTile
                  label="Intent κ"
                  value={fmtNum(portfolio.intent_eval?.cohen_kappa, 2)}
                />
              </div>
            ) : (
              <Loading />
            )}
          </section>

          <div className="workspace">
            <Sidebar
              accounts={accounts}
              selected={selected}
              onSelect={setSelected}
              filter={filter}
              setFilter={setFilter}
            />
            <main className="main">
              {selected ? (
                <AccountDetail accountId={selected} onLearned={refreshPortfolio} />
              ) : (
                <Loading label="Select an account" />
              )}
            </main>
          </div>
        </>
      )}

      {tab === "methods" && (
        <div className="workspace methods-wrap">
          <Methods />
        </div>
      )}

      <footer className="footer muted sm">
        Synthetic data · churn (XGBoost + SHAP) · LLM intent/sentiment + Cohen's κ · DoWhy causal
        layer · NBA agent with approval-gated memory · Langfuse tracing
      </footer>
    </div>
  );
}
