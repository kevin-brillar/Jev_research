"""Tool registry: the capabilities a node may be given, plus their instructions.

Tools are code, but everything *about* them that a model or a policy needs is
data attached to the registry entry:

* `instruction`  - the snippet appended to a node's prompt when the tool is
                   offered (mirrors Atenxion's `extract_tool_related_instruction`)
* `mutating`     - whether calling it changes state outside our records
* `limits`       - the numeric envelope the authorization gate judges against
* `request_markers` - words in a request that make this tool plausible, used by
                   deterministic action preparation instead of an if-chain per tool

Adding a capability is therefore one registry entry plus a line in a workflow
spec; no node, gate or policy code changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]
    instruction: str = ""
    mutating: bool = False
    limits: dict[str, Any] = field(default_factory=dict)
    request_markers: tuple[str, ...] = ()
    requires_authentication: bool = False


# ---------------------------------------------------------------- read-only tools


def knowledge_search(query: str = "") -> dict[str, Any]:
    """Search the published knowledge base. Offline stand-in for the RAG tool."""
    return {
        "query": query,
        "passages": [
            {
                "id": "policy-1.3",
                "text": "Refunds are available within 30 days of purchase. Beyond that window an exception needs approval by a person.",
            },
            {
                "id": "policy-2.1",
                "text": "Duplicate charges are corrected by refunding the duplicated amount once the account record confirms two identical charges.",
            },
        ],
    }


def account_lookup(customer_id: str = "self") -> dict[str, Any]:
    """Read the account record for this customer. Read-only."""
    return {
        "customer_id": customer_id,
        "subscription": "active",
        "last_invoice": {"number": "INV-2291", "amount": 120.0, "status": "settled"},
        "recent_charges": [
            {"id": "ch_8811", "amount": 120.0, "date": "2026-10-09", "status": "settled"},
            {"id": "ch_8812", "amount": 120.0, "date": "2026-10-09", "status": "settled"},
        ],
        "open_tickets": 0,
    }


def document_extract(file_id: str = "") -> dict[str, Any]:
    """Extract the values the customer asked about from an attached document."""
    return {
        "file_id": file_id,
        "fields": {"total": 1240.0, "reference": "INV-2291", "page": 1},
    }


def action_status(action_key: str) -> dict[str, Any]:
    """Read the journal entry for one action attempt. Read-only."""
    return _JOURNAL.get(action_key) or {"action_key": action_key, "status": "unknown"}


# ------------------------------------------------------------- mutating tools


_JOURNAL: dict[str, dict[str, Any]] = {}


def _key(tool: str, arguments: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{tool}:{digest}"


def refund_payment(amount: float, reason: str = "", idempotency_key: str | None = None) -> dict[str, Any]:
    """Refund a payment. Idempotent: the same key returns the first result."""
    key = idempotency_key or _key("refund_payment", {"amount": amount, "reason": reason})
    if key in _JOURNAL:
        return {**_JOURNAL[key], "idempotent_replay": True}
    record = {
        "action_key": key,
        "tool": "refund_payment",
        "status": "completed",
        "external_confirmation": True,
        "provider_reference": "re_2291_refund_01",
        "amount": amount,
        "reason": reason,
    }
    _JOURNAL[key] = record
    return record


def create_ticket(summary: str, idempotency_key: str | None = None) -> dict[str, Any]:
    """Open a support ticket. Idempotent on the same key."""
    key = idempotency_key or _key("create_ticket", {"summary": summary})
    if key in _JOURNAL:
        return {**_JOURNAL[key], "idempotent_replay": True}
    record = {
        "action_key": key,
        "tool": "create_ticket",
        "status": "completed",
        "external_confirmation": True,
        "provider_reference": "tk_5581",
        "summary": summary,
    }
    _JOURNAL[key] = record
    return record


def notify_customer(message: str, idempotency_key: str | None = None) -> dict[str, Any]:
    """Send the customer a message. Idempotent on the same key."""
    key = idempotency_key or _key("notify_customer", {"message": message})
    if key in _JOURNAL:
        return {**_JOURNAL[key], "idempotent_replay": True}
    record = {
        "action_key": key,
        "tool": "notify_customer",
        "status": "completed",
        "external_confirmation": True,
        "provider_reference": "msg_7712",
        "message": message,
    }
    _JOURNAL[key] = record
    return record


def journal() -> dict[str, dict[str, Any]]:
    """Inspect the in-process action journal (tests, demos)."""
    return dict(_JOURNAL)


def reset_journal() -> None:
    _JOURNAL.clear()


# ------------------------------------------------------------------ registry


TOOLS: dict[str, ToolDef] = {
    "knowledge_search": ToolDef(
        name="knowledge_search",
        description="Search published policy and product documentation.",
        fn=knowledge_search,
        instruction="Call knowledge_search before answering a documentation question, and cite the passage id in your answer.",
    ),
    "account_lookup": ToolDef(
        name="account_lookup",
        description="Read this customer's account, order and billing records.",
        fn=account_lookup,
        instruction="Call account_lookup to read account data. Report only what the record contains.",
        request_markers=("account", "balance", "invoice", "subscription", "order"),
    ),
    "document_extract": ToolDef(
        name="document_extract",
        description="Read the values the customer asked about out of the attached document.",
        fn=document_extract,
        instruction="Call document_extract to read the attachment, and name the page each value came from.",
        request_markers=("attached", "attachment", "document", "pdf", "contract", "invoice copy"),
    ),
    "action_status": ToolDef(
        name="action_status",
        description="Read the journal entry for one action attempt.",
        fn=action_status,
        instruction="Call action_status before describing the state of any action.",
    ),
    "refund_payment": ToolDef(
        name="refund_payment",
        description="Refund a payment to the customer.",
        fn=refund_payment,
        instruction=(
            "refund_payment is a mutating tool. Propose it as JSON; never describe it as done. "
            "It is executed by the action node after the gates pass."
        ),
        mutating=True,
        limits={"max_amount": 500.0, "requires_confirmation": True},
        request_markers=("refund", "charge", "charged", "money back"),
    ),
    "create_ticket": ToolDef(
        name="create_ticket",
        description="Open a support ticket for a person to pick up.",
        fn=create_ticket,
        instruction="create_ticket is mutating but reversible: it opens a ticket for a person rather than changing the account.",
        mutating=True,
        limits={"max_open_tickets": 5},
        request_markers=("ticket", "escalate", "speak to", "human"),
    ),
    "notify_customer": ToolDef(
        name="notify_customer",
        description="Send the customer a message about their case.",
        fn=notify_customer,
        instruction="notify_customer is mutating: it reaches the customer. Propose it, do not claim it happened.",
        mutating=True,
        limits={"max_messages_per_case": 3},
        request_markers=("email me", "notify", "let me know"),
    ),
}


def resolve(names: list[str] | None) -> list[ToolDef]:
    """Registry entries for a node's declared tools, failing loudly on typos."""
    resolved: list[ToolDef] = []
    for name in names or []:
        if name not in TOOLS:
            raise KeyError(f"Unknown tool '{name}'. Known: {sorted(TOOLS)}")
        resolved.append(TOOLS[name])
    return resolved


def callables(names: list[str] | None) -> list[Callable[..., dict[str, Any]]]:
    return [tool.fn for tool in resolve(names)]


def tool_instruction_block(names: list[str] | None) -> str:
    """Prompt-area block for the tools a node is offered (data, not code)."""
    snippets = [tool.instruction for tool in resolve(names) if tool.instruction]
    if not snippets:
        return ""
    lines = "\n".join(f"- {snippet}" for snippet in snippets)
    return f"\n\nAdditional Tool Specific Instructions:\n{lines}"


def mutating_tools(names: list[str] | None) -> list[ToolDef]:
    return [tool for tool in resolve(names) if tool.mutating]


def limits_for(tool: str) -> dict[str, Any]:
    definition = TOOLS.get(tool)
    return dict(definition.limits) if definition else {}


def request_markers_for(tool: str) -> tuple[str, ...]:
    definition = TOOLS.get(tool)
    return definition.request_markers if definition else ()
