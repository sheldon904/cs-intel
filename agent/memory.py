"""Hybrid memory: structured decision policy + semantic recall.

"Hybrid" because it combines three signals the agent uses on every proposal:

1. **Structured policy** — Laplace-smoothed approval rates per (action_type,
   risk_tier). This is the *learning* substrate: as operators approve/reject,
   the agent's ranking of which action to propose shifts measurably.
2. **Semantic recall** — a deterministic local embedding (hashing bag-of-words,
   no network) retrieves the most similar past decisions to ground the proposal.
3. **Recency** — recall is tie-broken toward recent decisions.

Everything persists to ``memory_store/decisions.jsonl`` so learning survives
restarts. No API key required.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.paths import MEMORY_DECISIONS_JSONL, ensure_dirs

EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hashing bag-of-words embedding (L2-normalized).

    Uses md5 rather than Python's ``hash`` because the built-in is salted per
    process — determinism matters so recall is reproducible across restarts.
    """
    vec = np.zeros(dim, dtype=float)
    for tok in _tokenize(text):
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are already normalized


@dataclass(slots=True)
class RecalledDecision:
    account_id: str
    action_type: str
    decision: str
    operator_note: str
    similarity: float
    created_at: str


class HybridMemory:
    """Append-only store of operator decisions with policy + semantic recall."""

    def __init__(self, path: Path = MEMORY_DECISIONS_JSONL) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = []
        self._embeddings: list[np.ndarray] = []
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        self._records.clear()
        self._embeddings.clear()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._records.append(rec)
                self._embeddings.append(embed(rec.get("context_text", "")))

    def add_decision(
        self,
        *,
        account_id: str,
        action_type: str,
        risk_tier: str,
        segment: str,
        decision: str,
        operator_note: str,
        context_text: str,
        created_at: str,
    ) -> None:
        ensure_dirs()
        rec = {
            "account_id": account_id,
            "action_type": action_type,
            "risk_tier": risk_tier,
            "segment": segment,
            "decision": decision,
            "operator_note": operator_note,
            "context_text": context_text,
            "created_at": created_at,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._records.append(rec)
        self._embeddings.append(embed(context_text))

    # -- structured policy --------------------------------------------------
    def approval_rate(self, action_type: str, risk_tier: str | None = None) -> float:
        """Laplace-smoothed P(approve | action_type[, risk_tier])."""
        approvals = rejections = 0
        for rec in self._records:
            if rec["action_type"] != action_type:
                continue
            if risk_tier is not None and rec["risk_tier"] != risk_tier:
                continue
            if rec["decision"] == "approved":
                approvals += 1
            elif rec["decision"] == "rejected":
                rejections += 1
        return (approvals + 1) / (approvals + rejections + 2)

    def policy_stats(self) -> dict[str, Any]:
        """Approval counts + rates per action type and per risk tier."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"approved": 0, "rejected": 0})
        by_tier: dict[str, dict[str, int]] = defaultdict(lambda: {"approved": 0, "rejected": 0})
        for rec in self._records:
            d = rec["decision"]
            if d in ("approved", "rejected"):
                counts[rec["action_type"]][d] += 1
                by_tier[f"{rec['risk_tier']}::{rec['action_type']}"][d] += 1
        rates = {
            at: {
                **c,
                "approval_rate": round((c["approved"] + 1) / (c["approved"] + c["rejected"] + 2), 3),
            }
            for at, c in counts.items()
        }
        return {"total_decisions": len(self._records), "by_action": rates}

    def rank_actions(self, action_types: list[str], risk_tier: str) -> list[str]:
        """Order candidate actions by learned approval rate for this risk tier."""
        return sorted(action_types, key=lambda a: self.approval_rate(a, risk_tier), reverse=True)

    # -- semantic recall ----------------------------------------------------
    def recall_similar(self, context_text: str, k: int = 3) -> list[RecalledDecision]:
        if not self._records:
            return []
        q = embed(context_text)
        sims = [_cosine(q, e) for e in self._embeddings]
        order = np.argsort(sims)[::-1][:k]
        out: list[RecalledDecision] = []
        for i in order:
            if sims[i] <= 0:
                continue
            rec = self._records[i]
            out.append(
                RecalledDecision(
                    account_id=rec["account_id"],
                    action_type=rec["action_type"],
                    decision=rec["decision"],
                    operator_note=rec.get("operator_note", ""),
                    similarity=round(float(sims[i]), 3),
                    created_at=rec.get("created_at", ""),
                )
            )
        return out

    def __len__(self) -> int:
        return len(self._records)
