"""Traced, provider-agnostic LLM client with a deterministic offline fallback.

Design goals
------------
1. **Runs with no secrets.** Every call takes a ``mock`` callable that produces
   the deterministic response used when no provider key is configured. This is
   what lets the intent labeler and the agent run inside ``docker compose up``
   with reproducible output and zero cost.
2. **Every call is traced.** Usage (tokens, latency, estimated cost) is appended
   to ``artifacts/llm_usage.jsonl`` for the local LLMOps summary, and mirrored to
   Langfuse when keys are present — satisfying "Langfuse tracing on every LLM
   call: labeling, agent proposals, evals".
3. **JSON-first.** Most callers want structured output; ``complete`` will parse
   and repair light JSON formatting issues before returning.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from common.config import Settings, get_settings
from common.paths import LLM_USAGE_JSONL, ensure_dirs

# Approximate public prices in USD per 1M tokens (input, output). Used both for
# real-call cost accounting and for the *projected* cost we surface in mock mode
# so the LLMOps dashboard shows a meaningful "cost per account" number offline.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mock": (1.00, 5.00),  # price mock tokens as if they were Haiku
}
_DEFAULT_PRICE = (1.00, 5.00)


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token)."""
    return max(1, round(len(text) / 4))


def _price_for(model: str) -> tuple[float, float]:
    return _PRICE_TABLE.get(model, _DEFAULT_PRICE)


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _price_for(model)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


@dataclass(slots=True)
class LLMResponse:
    text: str
    data: dict[str, Any] | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    projected_cost_usd: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object out of an LLM response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Grab the outermost {...} span as a fallback.
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class LLMClient:
    """Single entry point for all LLM calls in the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._anthropic: Any | None = None
        self._openai: Any | None = None
        self._langfuse: Any | None = None
        self._langfuse_ready = False

    # -- provider lazy loaders ---------------------------------------------
    def _anthropic_client(self) -> Any:
        if self._anthropic is None:
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._anthropic

    def _openai_client(self) -> Any:
        if self._openai is None:
            import openai

            self._openai = openai.OpenAI(api_key=self.settings.openai_api_key)
        return self._openai

    def _langfuse_client(self) -> Any | None:
        if self._langfuse_ready:
            return self._langfuse
        self._langfuse_ready = True
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception:  # pragma: no cover - tracing must never break the pipeline
            self._langfuse = None
        return self._langfuse

    # -- public API ---------------------------------------------------------
    def complete(
        self,
        *,
        name: str,
        system: str,
        user: str,
        mock: Callable[[str, str], str],
        stage: str = "unknown",
        json_mode: bool = True,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Run one completion, trace it, and (optionally) parse JSON.

        Parameters
        ----------
        name: short label for the call (shown in Langfuse).
        mock: deterministic ``(system, user) -> raw_text`` used in offline mode.
        stage: coarse pipeline stage ("intent_labeling", "agent_proposal",
            "intent_eval"...) used to group cost/usage in the LLMOps summary.
        """
        provider = self.settings.effective_provider
        metadata = metadata or {}
        t0 = time.perf_counter()

        if provider == "mock":
            text = mock(system, user)
            model = "mock"
            in_tok = estimate_tokens(system + user)
            out_tok = estimate_tokens(text)
        elif provider == "anthropic":
            text, model, in_tok, out_tok = self._call_anthropic(
                system, user, max_tokens, temperature
            )
        else:
            text, model, in_tok, out_tok = self._call_openai(
                system, user, max_tokens, temperature, json_mode
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        actual_cost = 0.0 if provider == "mock" else _cost_usd(model, in_tok, out_tok)
        # Projected cost prices even mock tokens, so the dashboard has real numbers.
        projected_cost = _cost_usd(
            self.settings.anthropic_model if provider == "mock" else model, in_tok, out_tok
        )
        data = _extract_json(text) if json_mode else None

        resp = LLMResponse(
            text=text,
            data=data,
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=actual_cost,
            projected_cost_usd=projected_cost,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        self._record_usage(name=name, stage=stage, resp=resp)
        self._trace_langfuse(
            name=name, stage=stage, system=system, user=user, resp=resp, metadata=metadata
        )
        return resp

    # -- provider implementations ------------------------------------------
    def _call_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> tuple[str, str, int, int]:
        client = self._anthropic_client()
        model = self.settings.anthropic_model
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return text, model, msg.usage.input_tokens, msg.usage.output_tokens

    def _call_openai(
        self, system: str, user: str, max_tokens: int, temperature: float, json_mode: bool
    ) -> tuple[str, str, int, int]:
        client = self._openai_client()
        model = self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, model, usage.prompt_tokens, usage.completion_tokens

    # -- observability ------------------------------------------------------
    def _record_usage(self, *, name: str, stage: str, resp: LLMResponse) -> None:
        ensure_dirs()
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "name": name,
            "stage": stage,
            "provider": resp.provider,
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "projected_cost_usd": resp.projected_cost_usd,
            "latency_ms": round(resp.latency_ms, 2),
            "account_id": resp.metadata.get("account_id"),
        }
        with LLM_USAGE_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _trace_langfuse(
        self,
        *,
        name: str,
        stage: str,
        system: str,
        user: str,
        resp: LLMResponse,
        metadata: dict[str, Any],
    ) -> None:
        lf = self._langfuse_client()
        if lf is None:
            return
        try:  # pragma: no cover - network path, exercised only with keys
            trace = lf.trace(name=stage, metadata=metadata)
            gen = trace.generation(
                name=name,
                model=resp.model,
                input={"system": system, "user": user},
                metadata={"stage": stage, **metadata},
            )
            gen.end(
                output=resp.text,
                usage={
                    "input": resp.input_tokens,
                    "output": resp.output_tokens,
                    "unit": "TOKENS",
                    "input_cost": resp.cost_usd,
                },
            )
        except Exception:
            pass  # tracing failures must never break the pipeline

    def flush(self) -> None:
        import contextlib

        lf = self._langfuse_client()
        if lf is not None:  # pragma: no cover
            with contextlib.suppress(Exception):
                lf.flush()


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide client singleton."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def usage_row_to_dict(resp: LLMResponse) -> dict[str, Any]:
    return asdict(resp)
