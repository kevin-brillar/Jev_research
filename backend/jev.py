"""The decision core: one Jev consultation, one verdict, one ledger entry.

`decide()` is the only place that knows how to go from "questions + state" to an
execution mode. Nodes never call a provider directly, so every path shares the
same fail-closed behaviour, the same policy engine and the same audit record.

`route_with_jev` keeps the signature the original scaffold used (and that the
Atenxion seam map treats as the integration contract) while becoming
config-driven: question set, provider and policy resolve from config or are
injected by tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend import config
from backend.answers import Answers, IncompleteAnswers
from backend.policy import PolicyEngine, Verdict
from backend.providers import Provider, build_provider
from backend.questions import QuestionSet, get_question_set

AUTOMATIC = "automatic"
REVIEW = "human_review"


class DecisionError(RuntimeError):
    """The consultation failed and the caller must fail closed."""


@dataclass
class Decision:
    """Everything one Jev consultation produced, ready for logs and prompts."""

    question_set: str
    route: str
    execution_mode: str
    confidence: float = 0.0
    risk_score: float = 0.0
    human_probability: float = 0.0
    reasons: list[str] = field(default_factory=list)
    provider: str = "mock"
    model: str = "mock/jev"
    request_id: str | None = None
    policy_rule: str = ""
    latency_ms: float = 0.0
    state_hash: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    raw_answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None

    @property
    def automatic(self) -> bool:
        return self.execution_mode == AUTOMATIC

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_set": self.question_set,
            "recommended_route": self.route,
            "execution_mode": self.execution_mode,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "human_probability": round(self.human_probability, 4),
            "reasons": list(self.reasons),
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "policy_rule": self.policy_rule,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }

    def prompt_vars(self) -> dict[str, Any]:
        """Keys referenced by guidance blocks as {jev_*} in config/guidance.yaml."""
        return {
            "jev_question_set": self.question_set,
            "jev_route": self.route,
            "jev_execution_mode": self.execution_mode,
            "jev_confidence": round(self.confidence, 3),
            "jev_risk": round(self.risk_score, 3),
            "jev_human_probability": round(self.human_probability, 3),
            "jev_reasons": "; ".join(self.reasons) if self.reasons else "no reasons recorded",
            "jev_model": self.model,
        }


# --------------------------------------------------------------------------- ledger


class DecisionLedger:
    """Append-only JSONL audit trail: one line per decision, never raw customer text."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("JEV_LEDGER_PATH")
        self.path = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[1] / "logs" / "jev_decisions.jsonl"
        )

    def append(self, decision: Decision, *, state_keys: list[str] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **decision.as_dict(),
            "state_hash": decision.state_hash,
            "state_keys": state_keys or [],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- core


def state_hash(state: Any) -> str:
    try:
        canonical = json.dumps(state, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(state)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def scope_state(state: dict[str, Any], state_scope: list[str] | None) -> dict[str, Any]:
    """Return the slice of state a node declared it needs.

    Scoping is the cheapest accuracy lever there is: a decision asked about one
    record is judged on that record, and a verifier that never receives the
    resolver's reasoning cannot inherit it. Absent paths are omitted, not nulled.
    """
    if not state_scope:
        return dict(state)

    scoped: dict[str, Any] = {}
    for path in state_scope:
        value: Any = state
        for part in path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if value is None:
            continue
        parts = path.split(".")
        target = scoped
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return scoped


async def decide(
    question_set: str | QuestionSet,
    state: dict[str, Any],
    *,
    policy: PolicyEngine | None = None,
    provider: Provider | None = None,
    ledger: DecisionLedger | None = None,
    mode: str | None = None,
) -> Decision:
    """Consult Jev once and return a verdict, failing closed on any failure."""
    resolved_set = get_question_set(question_set) if isinstance(question_set, str) else question_set
    engine = policy or PolicyEngine()
    active_provider = provider or build_provider(mode)
    started = time.perf_counter()
    digest = state_hash(state)

    try:
        answers: Answers = await active_provider.answer(resolved_set, state)
    except (IncompleteAnswers, KeyError, ValueError) as exc:
        return _failed(engine, resolved_set.name, f"{type(exc).__name__}: {exc}", active_provider, digest, started, ledger)
    except Exception as exc:  # provider transport, auth, timeout: still fail closed
        return _failed(engine, resolved_set.name, f"{type(exc).__name__}: {exc}", active_provider, digest, started, ledger)

    verdict: Verdict = engine.evaluate(resolved_set.name, answers)
    decision = Decision(
        question_set=resolved_set.name,
        route=verdict.route,
        execution_mode=verdict.execution_mode,
        confidence=answers.confidence,
        risk_score=answers.risk,
        human_probability=answers.human,
        reasons=list(verdict.reasons),
        provider=type(active_provider).__name__,
        model=answers.model,
        request_id=answers.request_id,
        policy_rule=verdict.rule_source,
        latency_ms=(time.perf_counter() - started) * 1000,
        state_hash=digest,
        probabilities=answers.probabilities,
        raw_answers=answers.by_question,
    )
    if ledger is not None:
        ledger.append(decision, state_keys=sorted(state))
    return decision


def _failed(
    engine: PolicyEngine,
    set_name: str,
    error: str,
    provider: Provider,
    digest: str,
    started: float,
    ledger: DecisionLedger | None,
) -> Decision:
    verdict = engine.fail_closed(set_name, error)
    decision = Decision(
        question_set=set_name,
        route=REVIEW,
        execution_mode=verdict.execution_mode,
        reasons=list(verdict.reasons),
        provider=type(provider).__name__,
        model=getattr(provider, "model", "unknown"),
        policy_rule=verdict.rule_source,
        latency_ms=(time.perf_counter() - started) * 1000,
        state_hash=digest,
        error=error,
    )
    if ledger is not None:
        ledger.append(decision)
    return decision


# --------------------------------------------------------------- preserved contract


async def route_with_jev(
    query: str,
    authenticated: bool = False,
    attachment_names: list[str] | None = None,
    *,
    question_set: str | None = None,
    policy: PolicyEngine | None = None,
    provider: Provider | None = None,
    ledger: DecisionLedger | None = None,
) -> dict[str, Any]:
    """Route and safety-check one Atenxion request, keeping the original signature.

    The question set defaults to JEV_QUESTION_SET (default "intake"), so callers
    that only need triage are unchanged while a new decision surface needs no new
    function.
    """
    state = {
        "request": {
            "text": query,
            "authenticated": authenticated,
            "attachment_names": attachment_names or [],
        }
    }
    decision = await decide(
        question_set or os.getenv("JEV_QUESTION_SET", "intake"),
        state,
        policy=policy,
        provider=provider,
        ledger=ledger,
    )
    return decision.as_dict()


def reload_config() -> None:
    """Drop cached config documents (tests, or after editing config)."""
    config.clear_caches()
