"""Tests for HybridMemory: policy learning + semantic recall + embeddings.

Every HybridMemory is constructed with a tmp_path so tests never read or write
the real ``agent/memory_store``.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.memory import HybridMemory, embed


@pytest.fixture
def memory(tmp_path):
    return HybridMemory(path=tmp_path / "decisions.jsonl")


def _add(mem, action_type, decision, *, risk_tier="High", account_id="ACC-0001",
         context_text="context", note="", created_at="2024-01-01T00:00:00+00:00"):
    mem.add_decision(
        account_id=account_id,
        action_type=action_type,
        risk_tier=risk_tier,
        segment="SMB",
        decision=decision,
        operator_note=note,
        context_text=context_text,
        created_at=created_at,
    )


def test_empty_memory_defaults_to_one_half(memory):
    # Laplace smoothing: no data -> P(approve) == 0.5
    assert memory.approval_rate("send_email") == 0.5
    assert memory.recall_similar("anything", k=3) == []
    assert len(memory) == 0


def test_approval_rate_rises_after_approvals(memory):
    for i in range(5):
        _add(memory, "send_email", "approved", created_at=f"2024-01-0{i + 1}T00:00:00+00:00")
    assert memory.approval_rate("send_email", "High") > 0.5


def test_approval_rate_falls_after_rejections(memory):
    for i in range(5):
        _add(memory, "schedule_call", "rejected", created_at=f"2024-02-0{i + 1}T00:00:00+00:00")
    assert memory.approval_rate("schedule_call", "High") < 0.5


def test_rank_actions_orders_by_approval_rate(memory):
    for i in range(5):
        _add(memory, "send_email", "approved", created_at=f"2024-01-0{i + 1}T00:00:00+00:00")
    for i in range(5):
        _add(memory, "schedule_call", "rejected", created_at=f"2024-02-0{i + 1}T00:00:00+00:00")

    ranked = memory.rank_actions(["schedule_call", "send_email"], "High")
    assert ranked == ["send_email", "schedule_call"]


def test_recall_similar_returns_most_similar_record(memory):
    _add(memory, "send_email", "approved",
         context_text="high risk cancel language renewal at risk", account_id="ACC-A")
    _add(memory, "schedule_call", "approved",
         context_text="expansion signal wants more seats upgrade", account_id="ACC-B")

    recalled = memory.recall_similar("cancel risk renewal language", k=1)
    assert len(recalled) == 1
    assert recalled[0].account_id == "ACC-A"
    assert recalled[0].action_type == "send_email"
    assert 0.0 < recalled[0].similarity <= 1.0


def test_policy_stats_counts_decisions(memory):
    for i in range(3):
        _add(memory, "send_email", "approved", created_at=f"2024-01-0{i + 1}T00:00:00+00:00")
    _add(memory, "send_email", "rejected", created_at="2024-01-09T00:00:00+00:00")

    stats = memory.policy_stats()
    assert stats["total_decisions"] == 4
    assert "send_email" in stats["by_action"]
    assert stats["by_action"]["send_email"]["approved"] == 3
    assert stats["by_action"]["send_email"]["rejected"] == 1


def test_memory_persists_across_reload(tmp_path):
    path = tmp_path / "decisions.jsonl"
    mem = HybridMemory(path=path)
    _add(mem, "send_email", "approved")
    # a fresh instance reading the same file recovers the record
    reloaded = HybridMemory(path=path)
    assert len(reloaded) == 1
    assert reloaded.approval_rate("send_email") > 0.5


def test_embed_is_deterministic_and_l2_normalized():
    v1 = embed("high risk cancel renewal")
    v2 = embed("high risk cancel renewal")
    assert np.array_equal(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0)

    # different text yields a different vector
    v3 = embed("expansion upgrade more seats")
    assert not np.array_equal(v1, v3)


def test_embed_empty_text_is_zero_vector():
    v = embed("")
    assert np.linalg.norm(v) == 0.0


def test_build_context_text_summarizes_state():
    from agent.nba_agent import build_context_text

    state = {
        "segment": "Enterprise",
        "risk_tier": "High",
        "top_factors": [
            {"feature": "cancel_risk_rate", "direction": "increases risk"},
            {"feature": "avg_adoption", "direction": "decreases risk"},
        ],
        "recent_tickets": [{"intent": "cancel-risk"}, {"intent": "bug-report"}],
    }
    text = build_context_text(state)
    assert "Enterprise" in text
    assert "High" in text
    # ticket intents are surfaced for semantic recall
    assert "cancel-risk" in text
