"""Deterministic node functions: the parts of a workflow that must not be an LLM.

Every function follows one contract so both runtimes can wrap it identically:

    async def fn(state: dict, node_input: Any, **params) -> NodeResult

`state` is the live session state (ADK 2 passes `ctx.state`, ADK 1 the session
state dict), `node_input` is whatever the previous node emitted, and `params` comes
from the node's `params:` block in the workflow spec. A NodeResult carries the
output for the next node, an optional route key, and the state updates the runtime
should apply.

Deterministic policy lives here rather than in a prompt: whether an action is
inside its tool's declared limits, whether a draft passes lint, how much a shadow
evaluation agrees with its labels. Judgments that need language live in LLM nodes,
and judgments that need calibrated probabilities live in Jev nodes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend import config, tools
from backend.answers import IncompleteAnswers
from backend.jev import Decision, decide
from backend.providers import FixtureProvider, MockProvider
from backend.questions import lint_question_set, parse_question_set

try:  # YAML is a dependency, but a function node should not hard-fail without it
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class NodeResult:
    output: Any = None
    route: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True


NodeFn = Callable[..., Awaitable[NodeResult]]


# --------------------------------------------------------------------- helpers


def _as_dict(value: Any) -> dict[str, Any]:
    """Accept a dict, a JSON object, or a fenced JSON/YAML block from a model."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    fence = re.search(r"```(?:json|yaml)?\s*(?P<body>.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group("body").strip()
    for loader in (json.loads, yaml.safe_load if yaml else None):
        if loader is None:
            continue
        try:
            parsed = loader(text)
        except Exception:  # noqa: BLE001 - a bad draft is an expected input
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _lookup(state: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a state path.

    Session state is keyed flat with dotted names ("case.pending_action"), which
    keeps ADK's own `{key}` instruction templating working and makes scoping and
    auditing trivial. A nested mapping is still accepted for hand-built state in
    tests, so a path is looked up flat first and then walked.
    """
    if path in state:
        return state[path]
    value: Any = state
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _request_text(state: dict[str, Any]) -> str:
    return str(_lookup(state, "request.text", "") or "")


# ---------------------------------------------------------------- intake / action


async def intake_normalize(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Build the canonical request record from whatever the runtime handed us."""
    text = ""
    if isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, dict):
        text = str(node_input.get("text") or node_input.get("query") or "")
    if not text:
        text = _request_text(state)

    attachments = params.get("attachments") or _lookup(state, "request.attachment_names", []) or []
    request = {
        "text": text.strip(),
        "authenticated": bool(_lookup(state, "request.authenticated", False)),
        "attachment_names": list(attachments),
    }
    case = {
        "id": params.get("case_prefix", "case") + "-0001",
        "request_text": request["text"],
        "reply": "",
        "proposed_action": None,
        "pending_action": None,
        "action_record": None,
        "handoff": "",
        "events": [],
    }
    return NodeResult(
        output=request,
        state={
            "request.text": request["text"],
            "request.authenticated": request["authenticated"],
            "request.attachment_names": request["attachment_names"],
            "case.id": case["id"],
            "case.request_text": case["request_text"],
            "case.events": [],
            "approvals": {},
            "jev_history": [],
        },
        log=[f"intake: {len(request['text'])} chars, {len(request['attachment_names'])} attachments"],
    )


async def prepare_action(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Validate a proposed action against its tool's declared limits.

    The decision the authorization gate later sees is this record, not the raw
    conversation, and the numeric envelope comes from the tool registry rather than
    from a branch per tool.
    """
    proposal = _as_dict(_lookup(state, "case.proposed_action", node_input))
    tool_name = str(proposal.get("tool") or "")
    raw_arguments = proposal.get("arguments")
    arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, dict) else {}
    definition = tools.TOOLS.get(tool_name)

    if definition is None or tool_name not in (params.get("allowed_tools") or list(tools.TOOLS)):
        pending = {"tool": tool_name, "arguments": arguments, "mutating": False, "reason": "unknown tool", "valid": False}
        return NodeResult(
            output=pending,
            state={"case.pending_action": pending},
            log=[f"prepare_action: rejected unknown tool '{tool_name}'"],
        )

    limits = tools.limits_for(tool_name)
    max_amount = limits.get("max_amount")
    amount_value = arguments.get("amount")
    amount = float(amount_value) if isinstance(amount_value, (int, float)) else None
    within_limits = True
    reasons: list[str] = []
    if max_amount is not None and amount is not None and float(amount) > float(max_amount):
        within_limits = False
        reasons.append(f"amount {amount} exceeds the {max_amount} limit for {tool_name}")

    markers = tools.request_markers_for(tool_name)
    text = _request_text(state).lower()
    matches_request = not markers or any(marker in text for marker in markers)
    if not matches_request:
        reasons.append(f"the request text does not mention anything {tool_name} acts on")

    pending = {
        "tool": tool_name,
        "arguments": arguments,
        "mutating": definition.mutating,
        "limits": limits,
        "within_limits": within_limits,
        "matches_request": matches_request,
        "rationale": proposal.get("rationale", ""),
        "valid": within_limits and matches_request,
        "reasons": reasons,
    }
    return NodeResult(
        output=pending,
        state={"case.pending_action": pending},
        log=[f"prepare_action: {tool_name} valid={pending['valid']} {'; '.join(reasons)}".strip()],
    )


async def execute_action(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Perform exactly one approved action, idempotently, and record it.

    This is the only node in the workflow that causes an external side effect, so
    the record it writes is what the verifier and the ledger read.
    """
    pending = _lookup(state, "case.pending_action", node_input) or {}
    tool_name = str(pending.get("tool") or "")
    arguments = pending.get("arguments") or {}
    definition = tools.TOOLS.get(tool_name)

    if definition is None or not definition.mutating:
        record = {
            "tool": tool_name,
            "arguments": arguments,
            "status": "failed",
            "external_confirmation": False,
            "response": {"error": "no mutating tool resolved"},
        }
        return NodeResult(
            output=record,
            state={"case.action_record": record},
            log=["execute_action: refused, no mutating tool resolved"],
        )

    idempotency_key = str(params.get("case_id") or _lookup(state, "case.id", "case")) + ":" + tool_name
    try:
        response = definition.fn(**arguments, idempotency_key=idempotency_key)
    except TypeError as exc:  # arguments the tool does not accept
        response = {"status": "failed", "error": f"argument mismatch: {exc}"}

    record = {
        "tool": tool_name,
        "arguments": arguments,
        "idempotency_key": idempotency_key,
        "status": response.get("status", "unknown"),
        "external_confirmation": bool(response.get("external_confirmation", False)),
        "provider_reference": response.get("provider_reference"),
        "response": response,
    }
    events = list(_lookup(state, "case.events", []) or [])
    events.append({"node": "execute_action", "tool": tool_name, "status": record["status"]})
    return NodeResult(
        output=record,
        state={"case.action_record": record, "case.events": events},
        log=[f"execute_action: {tool_name} -> {record['status']} (confirmation={record['external_confirmation']})"],
    )


# ------------------------------------------------------------------ jev builder


async def builder_start(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Seed the builder state from the spec's defaults plus the incoming brief."""
    brief = _as_dict(node_input) or _lookup(state, "builder.brief", {}) or {}
    if isinstance(node_input, str) and node_input.strip():
        brief = {**brief, "description": node_input.strip()}
    builder = {
        "brief": brief,
        "attempts": 0,
        "max_attempts": int(params.get("max_attempts", _lookup(state, "builder.max_attempts", 3) or 3)),
        "min_agreement": float(params.get("min_agreement", _lookup(state, "builder.min_agreement", 1.0) or 1.0)),
        "requirements": None,
        "draft": None,
        "lint_report": None,
        "cases": None,
        "eval_report": None,
        "thresholds": None,
        "report": "",
        "rejection": "",
    }
    return NodeResult(output=brief, state={"builder": builder}, log=["builder_start: attempt budget set"])


async def lint_draft(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Lint the draft question set and decide whether to keep looping.

    The attempt budget is the loop bound: `exhausted` leaves the loop rather than
    looping forever, which is the failure mode an unbounded review cycle produces.
    """
    draft = _as_dict(_lookup(state, "builder.draft", node_input))
    attempts = int(_lookup(state, "builder.attempts", 0) or 0) + 1
    max_attempts = int(_lookup(state, "builder.max_attempts", params.get("max_attempts", 3)) or 3)

    report: dict[str, Any] = {"ok": False, "errors": [{"rule": "draft.unparsable", "message": "draft is not a mapping"}]}
    if draft:
        set_name = str(draft.get("set") or "draft")
        set_doc = {
            "description": draft.get("description", ""),
            "state_fields": draft.get("state_fields") or [],
            "questions": draft.get("questions") or {},
        }
        try:
            question_set = parse_question_set(set_name, set_doc, {"no_match_required": False}, "draft")
            report = lint_question_set(question_set).as_dict()
        except Exception as exc:  # noqa: BLE001 - a malformed draft is an expected input
            report = {"ok": False, "errors": [{"rule": "draft.invalid", "message": str(exc)}]}

    if report.get("ok"):
        route = "valid"
    elif attempts < max_attempts:
        route = "invalid"
    else:
        route = "exhausted"

    return NodeResult(
        output=report,
        route=route,
        state={"builder": {"attempts": attempts, "lint_report": report}},
        log=[f"lint_draft: attempt {attempts}/{max_attempts} ok={report.get('ok')} -> {route}"],
    )


async def shadow_evaluate(state: dict[str, Any], node_input: Any, **params: Any) -> NodeResult:
    """Replay the draft question set against labelled cases, offline.

    The provider is injectable (`params.provider`), because the number this node
    produces is only meaningful with the provider that will run in production:
    with the deterministic mock it measures the pipeline, with a live provider it
    measures the model.
    """
    cases = _as_dict(_lookup(state, "builder.cases", node_input)).get("cases") or []
    draft = _as_dict(_lookup(state, "builder.draft", {}))
    set_name = str(draft.get("set") or "draft")
    set_doc = {
        "description": draft.get("description", ""),
        "state_fields": draft.get("state_fields") or ["request.text"],
        "questions": draft.get("questions") or {},
    }

    provider_kind = str(params.get("provider", "mock"))
    if provider_kind == "fixture":
        fixture_answers = _lookup(state, "builder.fixture_answers", {}) or {}
        provider = FixtureProvider(answers=fixture_answers)
    else:
        provider = MockProvider()

    agrees = 0
    auto = 0
    results: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        expected = str(case.get("expected") or "")
        case_state = {"request": {"text": str(case.get("text") or ""), "authenticated": False, "attachment_names": []}}
        try:
            question_set = parse_question_set(set_name, set_doc, {"no_match_required": False}, "draft")
            decision: Decision = await decide(question_set, case_state, provider=provider)
        except (IncompleteAnswers, KeyError, ValueError) as exc:
            results.append({"id": case.get("id"), "expected": expected, "got": None, "error": str(exc)})
            continue
        matched = decision.route == expected
        agrees += int(matched)
        auto += int(decision.automatic)
        results.append(
            {
                "id": case.get("id"),
                "expected": expected,
                "got": decision.route,
                "confidence": decision.confidence,
                "execution_mode": decision.execution_mode,
                "match": matched,
            }
        )

    total = len([r for r in results if r.get("got") is not None]) or 1
    agreement = round(agrees / total, 4)
    report = {
        "set": set_name,
        "provider": provider_kind,
        "cases": len(cases),
        "scored": total,
        "agreement": agreement,
        "automatic_rate": round(auto / total, 4),
        "min_agreement": float(_lookup(state, "builder.min_agreement", params.get("min_agreement", 1.0)) or 1.0),
        "results": results,
    }
    route = "pass" if agreement >= report["min_agreement"] else "fail"
    return NodeResult(
        output=report,
        route=route,
        state={"builder": {"eval_report": report}},
        log=[f"shadow_evaluate: agreement {agreement} ({report['scored']} scored) -> {route}"],
    )


def fixture_provider(answers: dict[str, dict[str, Any]]) -> FixtureProvider:
    """Small helper so tests and notebooks can pin exact answers."""
    return FixtureProvider(answers=answers)


NODE_FUNCTIONS: dict[str, NodeFn] = {
    "intake_normalize": intake_normalize,
    "prepare_action": prepare_action,
    "execute_action": execute_action,
    "builder_start": builder_start,
    "lint_draft": lint_draft,
    "shadow_evaluate": shadow_evaluate,
}


def resolve_function(name: str, **params: Any) -> Callable[..., Awaitable[NodeResult]]:
    """Bind spec params to a registry function, failing loudly on a typo."""
    if name not in NODE_FUNCTIONS:
        raise KeyError(f"Unknown node function '{name}'. Known: {sorted(NODE_FUNCTIONS)}")
    fn = NODE_FUNCTIONS[name]

    async def bound(state: dict[str, Any], node_input: Any) -> NodeResult:
        return await fn(state, node_input, **params)

    bound.__name__ = name
    return bound


def config_bundle() -> dict[str, Any]:
    """The config documents a run needs, for diagnostics in tests and the UI."""
    return {
        "policy_sets": sorted(config.policy().get("sets", {})),
        "guidance_blocks": sorted(config.guidance().get("blocks", {})),
        "workflows": config.workflow_names(),
    }
