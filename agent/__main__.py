"""CLI demo: ``python -m agent`` runs a propose -> approve -> learn loop."""

from __future__ import annotations

from agent.nba_agent import NBAAgent
from api.repository import get_repository


def main() -> None:
    repo = get_repository()
    agent = NBAAgent()

    high_risk = [a for a in repo.list_accounts(risk_tier="High")][:3]
    if not high_risk:
        print("No high-risk accounts found. Run the pipeline first.")
        return

    print(f"Memory starts with {len(agent.memory)} decisions.\n")
    for a in high_risk:
        state = repo.get_account_detail(a["account_id"])
        assert state is not None
        proposal = agent.propose(state)
        print(f"== {state['company_name']} ({state['account_id']}) — {state['risk_tier']} risk ==")
        print(f"  Proposed: [{proposal.action_type}] {proposal.title}")
        print(f"  Priority: {proposal.priority}")
        print(f"  Rationale: {proposal.rationale}")
        print(f"  Memory: {proposal.memory_influence}")
        # Simulate an operator approving escalations but rejecting anything else.
        decision = "approved" if proposal.action_type == "escalation" else "rejected"
        agent.record_decision(proposal.proposal_id, decision, operator_note=f"demo: {decision}")
        print(f"  -> operator {decision}. Memory now has {len(agent.memory)} decisions.\n")

    print("Policy learned so far:")
    print(agent.memory.policy_stats())


if __name__ == "__main__":
    main()
