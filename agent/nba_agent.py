"""The next-best-action agent: propose -> human gate -> learn."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.actions import ACTION_LABELS, ActionType, Decision, Proposal
from agent.memory import HybridMemory
from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    _candidate_actions,
    build_agent_user_prompt,
    humanize_factor,
    mock_proposal,
)
from common.llm import LLMClient, get_client
from common.paths import PROPOSALS_JSONL, ensure_dirs

_VALID_ACTIONS = {a.value for a in ActionType}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_context_text(state: dict[str, Any]) -> str:
    """Compact text summary used for semantic memory recall."""
    drivers = [
        humanize_factor(f["feature"])
        for f in state.get("top_factors", [])
        if f.get("direction") == "increases risk"
    ][:4]
    intents = [t["intent"] for t in state.get("recent_tickets", [])]
    return (
        f"{state.get('segment')} {state['risk_tier']} risk; "
        f"drivers: {', '.join(drivers)}; ticket intents: {', '.join(intents)}"
    )


class ProposalStore:
    """Append-only JSONL of every proposal (pending + resolved)."""

    def __init__(self, path: Path = PROPOSALS_JSONL) -> None:
        self.path = path

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write_all(self, rows: list[dict[str, Any]]) -> None:
        ensure_dirs()
        with self.path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def add(self, proposal: Proposal) -> None:
        ensure_dirs()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(proposal.to_dict()) + "\n")

    def get(self, proposal_id: str) -> Proposal | None:
        for r in self._read_all():
            if r["proposal_id"] == proposal_id:
                return Proposal.from_dict(r)
        return None

    def update(self, proposal: Proposal) -> None:
        rows = self._read_all()
        for i, r in enumerate(rows):
            if r["proposal_id"] == proposal.proposal_id:
                rows[i] = proposal.to_dict()
                break
        self._write_all(rows)

    def list(self, account_id: str | None = None) -> list[Proposal]:
        rows = self._read_all()
        props = [Proposal.from_dict(r) for r in rows]
        if account_id:
            props = [p for p in props if p.account_id == account_id]
        return sorted(props, key=lambda p: p.created_at, reverse=True)

    def latest_for(self, account_id: str) -> Proposal | None:
        props = self.list(account_id)
        return props[0] if props else None


class NBAAgent:
    def __init__(
        self, memory: HybridMemory | None = None, client: LLMClient | None = None
    ) -> None:
        self.memory = memory or HybridMemory()
        self.store = ProposalStore()
        self.client = client or get_client()

    def propose(self, state: dict[str, Any]) -> Proposal:
        """Generate + persist a pending proposal for an account."""
        context_text = build_context_text(state)
        recalled = [asdict(r) for r in self.memory.recall_similar(context_text, k=3)]
        candidates = _candidate_actions(state)
        ranked = self.memory.rank_actions(candidates, state["risk_tier"])

        resp = self.client.complete(
            name="nba_proposal",
            system=AGENT_SYSTEM_PROMPT,
            user=build_agent_user_prompt(state, recalled, ranked),
            mock=lambda _s, _u: mock_proposal(state, recalled, ranked),
            stage="agent_proposal",
            max_tokens=600,
            metadata={"account_id": state["account_id"]},
        )
        data = resp.data or {}
        action_type = data.get("action_type")
        if action_type not in _VALID_ACTIONS:
            data = json.loads(mock_proposal(state, recalled, ranked))
            action_type = data["action_type"]

        influence = self._describe_influence(ranked, recalled)
        proposal = Proposal(
            proposal_id=uuid.uuid4().hex[:12],
            account_id=state["account_id"],
            created_at=_now(),
            action_type=action_type,
            title=data.get("title", ACTION_LABELS.get(ActionType(action_type), "Action")),
            body=data.get("body", ""),
            rationale=data.get("rationale", ""),
            priority=data.get("priority", "medium"),
            churn_probability=float(state["churn_probability"]),
            risk_tier=state["risk_tier"],
            evidence=data.get("evidence", []),
            memory_influence=influence,
        )
        self.store.add(proposal)
        return proposal

    def record_decision(
        self, proposal_id: str, decision: str, operator_note: str = ""
    ) -> Proposal:
        """Apply the human approve/reject and write it back into memory."""
        if decision not in (Decision.APPROVED.value, Decision.REJECTED.value):
            raise ValueError(f"decision must be approved/rejected, got {decision!r}")
        proposal = self.store.get(proposal_id)
        if proposal is None:
            raise KeyError(f"unknown proposal {proposal_id}")

        proposal.decision = decision
        proposal.operator_note = operator_note
        proposal.resolved_at = _now()
        self.store.update(proposal)

        self.memory.add_decision(
            account_id=proposal.account_id,
            action_type=proposal.action_type,
            risk_tier=proposal.risk_tier,
            segment="",  # segment is optional in the record; kept for schema symmetry
            decision=decision,
            operator_note=operator_note,
            context_text=f"{proposal.risk_tier} risk; {proposal.title}; {proposal.rationale}",
            created_at=proposal.resolved_at,
        )
        return proposal

    def _describe_influence(
        self, ranked: list[str], recalled: list[dict[str, Any]]
    ) -> str:
        if len(self.memory) == 0:
            return "No prior decisions yet — using default policy for this risk tier."
        top = ranked[0] if ranked else "n/a"
        rate = self.memory.approval_rate(top)
        note = (
            f"Ranked '{top}' first (approval rate {rate:.0%} across {len(self.memory)} "
            f"past decisions)."
        )
        if recalled:
            note += f" Recalled {len(recalled)} similar case(s)."
        return note
