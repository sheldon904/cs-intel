"""Action taxonomy + proposal / decision data structures for the NBA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    EMAIL_DRAFT = "email_draft"          # a ready-to-send outreach email
    ESCALATION = "escalation"            # escalate to leadership / save team
    CSM_TASK = "csm_task"                # a task for the success manager
    EXPANSION_OUTREACH = "expansion_outreach"  # capitalize on an expansion signal


ACTION_LABELS = {
    ActionType.EMAIL_DRAFT: "Send outreach email",
    ActionType.ESCALATION: "Escalate to save team",
    ActionType.CSM_TASK: "Create success-manager task",
    ActionType.EXPANSION_OUTREACH: "Expansion outreach",
}


class Decision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(slots=True)
class Proposal:
    """A single proposed intervention, awaiting human approval."""

    proposal_id: str
    account_id: str
    created_at: str
    action_type: str
    title: str
    body: str  # email text or task description
    rationale: str
    priority: str  # "high" | "medium" | "low"
    churn_probability: float
    risk_tier: str
    evidence: list[str] = field(default_factory=list)  # bullet points shown to operator
    memory_influence: str = ""  # how past decisions shaped this proposal
    decision: str = Decision.PENDING.value
    operator_note: str = ""
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Proposal:
        return cls(**d)
