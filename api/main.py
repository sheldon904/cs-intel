"""FastAPI application wiring the pipeline artifacts + NBA agent into a dashboard API."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.nba_agent import NBAAgent
from api.repository import Repository, get_repository
from api.schemas import DecisionRequest
from common.config import get_settings
from common.paths import CALIBRATION_PNG, PR_CURVE_PNG, SHAP_SUMMARY_PNG

app = FastAPI(
    title="cs-intel API",
    version="0.1.0",
    description="Churn scoring, intent-tagged tickets, causal results, and an "
    "NBA agent with a human approval gate.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PLOTS = {
    "calibration": CALIBRATION_PNG,
    "shap": SHAP_SUMMARY_PNG,
    "pr": PR_CURVE_PNG,
}


def repo() -> Repository:
    return get_repository()


# A single agent instance keeps the in-memory learning hot within the process.
_agent: NBAAgent | None = None


def agent() -> NBAAgent:
    global _agent
    if _agent is None:
        _agent = NBAAgent()
    return _agent


def _memory_view() -> dict:
    a = agent()
    recent = [asdict(r) for r in a.memory.recall_similar("recent decision", k=5)]
    return {"policy": a.memory.policy_stats(), "recent_recall_sample": recent}


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": s.effective_provider,
        "langfuse_enabled": s.langfuse_enabled,
    }


@app.get("/api/portfolio")
def portfolio() -> dict:
    r = repo()
    metrics = r.churn_metrics()
    return {
        "summary": r.portfolio_summary(),
        "churn_auc": metrics.get("xgboost", {}).get("auc"),
        "intent_eval": r.intent_eval().get("intent", {}),
        "causal": {
            "true": r.causal_results().get("true_ate_risk_diff"),
            "naive": r.causal_results().get("naive_estimate"),
            "adjusted": r.causal_results().get("backdoor_gformula_logistic"),
        },
    }


@app.get("/api/accounts")
def list_accounts(risk_tier: str | None = None) -> list[dict]:
    return repo().list_accounts(risk_tier=risk_tier)


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str) -> dict:
    detail = repo().get_account_detail(account_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
    latest = agent().store.latest_for(account_id)
    detail["latest_proposal"] = latest.to_dict() if latest else None
    return detail


@app.post("/api/accounts/{account_id}/propose")
def propose(account_id: str) -> dict:
    detail = repo().get_account_detail(account_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
    proposal = agent().propose(detail)
    return {"proposal": proposal.to_dict(), "memory": _memory_view()}


@app.post("/api/proposals/{proposal_id}/decision")
def decide(proposal_id: str, req: DecisionRequest) -> dict:
    try:
        proposal = agent().record_decision(proposal_id, req.decision, req.note)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown proposal {proposal_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"proposal": proposal.to_dict(), "memory": _memory_view()}


@app.get("/api/proposals")
def list_proposals(account_id: str | None = None) -> list[dict]:
    return [p.to_dict() for p in agent().store.list(account_id=account_id)]


@app.get("/api/memory")
def memory() -> dict:
    return _memory_view()


@app.get("/api/metrics/churn")
def metrics_churn() -> dict:
    return repo().churn_metrics()


@app.get("/api/metrics/intents")
def metrics_intents() -> dict:
    return repo().intent_eval()


@app.get("/api/metrics/causal")
def metrics_causal() -> dict:
    return repo().causal_results()


@app.get("/api/metrics/llmops")
def metrics_llmops() -> dict:
    return repo().llmops_summary()


@app.get("/api/plots/{name}")
def plot(name: str) -> FileResponse:
    path = _PLOTS.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown plot {name}")
    return FileResponse(path, media_type="image/png")
