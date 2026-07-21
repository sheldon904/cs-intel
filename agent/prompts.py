"""Agent system prompt + deterministic offline proposal generator."""

from __future__ import annotations

import json
from typing import Any

from agent.actions import ActionType

AGENT_SYSTEM_PROMPT = """You are a customer-success strategist agent. Given an
at-risk account's state — churn risk, the model's SHAP risk drivers, recent
intent-tagged support tickets, and a memory of how operators handled similar
accounts — propose ONE next-best-action.

You never execute anything. A human reviews your proposal and approves or
rejects it. Favour action types that operators have historically approved for
similar accounts (this is provided to you). Be specific and reference the
account's actual signals.

Respond with ONLY a JSON object:
{"action_type": one of [email_draft, escalation, csm_task, expansion_outreach],
 "title": str, "body": str, "rationale": str,
 "priority": one of [high, medium, low], "evidence": [str, ...]}"""

# Map raw feature names to plain-language risk drivers for operators.
_FEATURE_PHRASES = {
    "nps_mean": "low NPS survey scores",
    "nps_last": "a low recent NPS score",
    "avg_adoption": "low feature adoption",
    "last_adoption": "declining feature adoption",
    "adoption_trend": "a downward adoption trend",
    "active_seat_ratio": "few active seats vs contracted",
    "avg_logins": "low login activity",
    "avg_feature_events": "low feature usage",
    "tickets_total": "a high volume of support tickets",
    "tickets_per_month": "frequent support tickets",
    "avg_sentiment": "negative ticket sentiment",
    "pct_months_with_tickets": "persistent support issues",
    "cancel_risk_rate": "cancellation-risk language in tickets",
    "intent_negative_rate": "negative-sentiment tickets",
    "expansion_signal_rate": "expansion signals in tickets",
    "confusion_rate": "confusion / how-to tickets",
    "bug_report_rate": "bug reports",
    "log_seats": "account size",
    "seats_contracted": "contracted seat count",
    "mrr": "monthly revenue",
    "tenure_months": "account tenure",
}


def humanize_factor(feature: str) -> str:
    return _FEATURE_PHRASES.get(feature, feature.replace("_", " "))


def _top_factor_phrases(state: dict[str, Any], k: int = 3) -> list[str]:
    phrases = []
    for f in state.get("top_factors", []):
        if f.get("direction") == "increases risk":
            phrases.append(humanize_factor(f["feature"]))
        if len(phrases) >= k:
            break
    return phrases


def build_agent_user_prompt(
    state: dict[str, Any], recalled: list[dict[str, Any]], ranked_actions: list[str]
) -> str:
    tickets = "\n".join(
        f'  - [{t["intent"]}/{t["sentiment"]}] {t["text"]}' for t in state.get("recent_tickets", [])[:5]
    )
    mem = (
        "\n".join(
            f'  - {r["account_id"]}: {r["action_type"]} was {r["decision"]}'
            f' (similarity {r["similarity"]})' + (f' — "{r["operator_note"]}"' if r.get("operator_note") else "")
            for r in recalled
        )
        or "  (no similar past decisions yet)"
    )
    return f"""Account: {state.get('company_name')} ({state['account_id']})
Segment: {state.get('segment')} | Seats: {state.get('seats_contracted')} | MRR: ${state.get('mrr')}
Churn risk: {state['churn_probability']:.0%} ({state['risk_tier']})
Renewal: {state.get('renewal_date')}

Top model risk drivers (SHAP):
{chr(10).join('  - ' + humanize_factor(f['feature']) + ' (' + f['direction'] + ')' for f in state.get('top_factors', [])[:5])}

Recent intent-tagged tickets:
{tickets or '  (none)'}

Operator-approved action ranking for this risk tier (learned from memory):
  {', '.join(ranked_actions)}

Similar past operator decisions:
{mem}

Propose the single next-best-action as JSON."""


# ---- deterministic offline proposal --------------------------------------
def _candidate_actions(state: dict[str, Any]) -> list[str]:
    risk = state["risk_tier"]
    intents = state.get("intent_summary", {})
    expansion = intents.get("expansion_signal_rate", 0.0)
    candidates: list[str]
    if risk == "High":
        candidates = [ActionType.ESCALATION.value, ActionType.EMAIL_DRAFT.value, ActionType.CSM_TASK.value]
    elif risk == "Medium":
        candidates = [ActionType.EMAIL_DRAFT.value, ActionType.CSM_TASK.value, ActionType.ESCALATION.value]
    else:
        candidates = [ActionType.CSM_TASK.value, ActionType.EMAIL_DRAFT.value]
    if expansion >= 0.30 and risk != "High":
        candidates = [ActionType.EXPANSION_OUTREACH.value, *candidates]
    return candidates


def _body_for(action_type: str, state: dict[str, Any], drivers: list[str]) -> tuple[str, str]:
    company = state.get("company_name", "the account")
    driver_txt = ", ".join(drivers) if drivers else "recent engagement signals"
    if action_type == ActionType.ESCALATION.value:
        title = f"Escalate {company} to the save team"
        body = (
            f"{company} is at {state['churn_probability']:.0%} churn risk ahead of renewal "
            f"({state.get('renewal_date')}). Drivers: {driver_txt}. Recommend the save team "
            f"engage the economic buyer this week with a value-recovery plan and a check-in cadence."
        )
    elif action_type == ActionType.EMAIL_DRAFT.value:
        title = f"Outreach email to {company}"
        body = (
            f"Hi {company} team,\n\nI wanted to personally check in. We've noticed {driver_txt} "
            f"and want to make sure you're getting full value ahead of your renewal. Could we grab "
            f"30 minutes this week to walk through your goals and unblock anything? — Your CS team"
        )
    elif action_type == ActionType.EXPANSION_OUTREACH.value:
        title = f"Expansion outreach to {company}"
        body = (
            f"{company} is showing expansion signals in recent tickets. Reach out with tailored "
            f"pricing for additional seats / the next tier and offer an enablement session for the "
            f"new team. Strike while engagement is high."
        )
    else:  # CSM_TASK
        title = f"Success-manager task for {company}"
        body = (
            f"Assign the CSM to review {company} ({driver_txt}). Schedule a health check, confirm "
            f"the champion is still active, and log a renewal-readiness note before "
            f"{state.get('renewal_date')}."
        )
    return title, body


def mock_proposal(
    state: dict[str, Any], recalled: list[dict[str, Any]], ranked_actions: list[str]
) -> str:
    """Deterministic proposal used offline. Picks the top learned-preferred action."""
    action_type = ranked_actions[0] if ranked_actions else ActionType.CSM_TASK.value
    drivers = _top_factor_phrases(state)
    title, body = _body_for(action_type, state, drivers)
    risk = state["risk_tier"]
    priority = "high" if risk == "High" else "medium" if risk == "Medium" else "low"
    evidence = [f"Churn risk {state['churn_probability']:.0%} ({risk})"]
    evidence += [f"Risk driver: {d}" for d in drivers[:2]]
    cancel = state.get("intent_summary", {}).get("cancel_risk_rate", 0.0)
    if cancel > 0:
        evidence.append(f"{cancel:.0%} of tickets flagged cancel-risk by the LLM labeler")
    return json.dumps(
        {
            "action_type": action_type,
            "title": title,
            "body": body,
            "rationale": (
                f"Selected '{action_type}' — the action operators have most often approved for "
                f"{risk}-risk accounts, given {', '.join(drivers) or 'the current signals'}."
            ),
            "priority": priority,
            "evidence": evidence,
        }
    )
