"""Instruction assembly: the prompt-area surface, assembled from config.

Mirrors the shape the Atenxion backend uses (`base_agent.py` builds
`full_instruction` from the node's instruction plus modular, conditionally
included guidance blocks, and `guardrails/instruction_hints.py` decides which
blocks apply). Here every block is data in config/guidance.yaml, a node selects
blocks by name in the workflow spec, and the tool block comes from the tool
registry.

Two rules this module exists to enforce:

1. No prompt text lives in Python. Assembled = spec instruction + tool
   instructions + selected guidance blocks + a runtime footer template, all of
   which come from config or the spec.
2. Anything a node needs to know about the run — including a Jev decision — is a
   `{state_key}` placeholder, so behaviour changes are config edits.
"""

from __future__ import annotations

import re
from typing import Any

from backend import config, tools

_TEMPLATE_RE = re.compile(r"\{(?P<key>[A-Za-z0-9_.\[\]]+)\}")

RUNTIME_FOOTER_FALLBACK = "Workflow {workflow_id} / node {node_id} / model {model}."


class UnknownGuidanceBlock(KeyError):
    """A node referenced a guidance block that guidance.yaml does not define."""


def guidance_blocks() -> dict[str, str]:
    return dict(config.guidance().get("blocks") or {})


def guidance_footer_template() -> str:
    templates = config.guidance().get("templates") or {}
    return str(templates.get("runtime_facts", RUNTIME_FOOTER_FALLBACK))


def select_blocks(names: list[str] | None, node_id: str) -> list[str]:
    """Resolve a node's declared blocks, failing loudly on a typo in the spec."""
    available = guidance_blocks()
    selected: list[str] = []
    for name in names or []:
        if name not in available:
            raise UnknownGuidanceBlock(
                f"node '{node_id}' asks for guidance block '{name}'; defined blocks: {sorted(available)}"
            )
        selected.append(name)
    return selected


def placeholders(text: str) -> list[str]:
    return sorted({match.group("key") for match in _TEMPLATE_RE.finditer(text)})


def assemble_instruction(
    *,
    node_id: str,
    instruction: str,
    workflow_id: str,
    model: str,
    guidance: list[str] | None = None,
    tools_offered: list[str] | None = None,
    extra_blocks: list[str] | None = None,
) -> str:
    """Build one node's full instruction from data.

    Order matters and mirrors the production backend: the node's own instruction
    first, then tool-specific instructions, then the guidance blocks, then a short
    footer naming the workflow, node and model (useful in traces).
    """
    blocks = guidance_blocks()
    parts: list[str] = [instruction.strip()]

    tool_block = tools.tool_instruction_block(tools_offered)
    if tool_block:
        parts.append(tool_block.strip())

    for name in select_blocks(guidance, node_id):
        parts.append(blocks[name].strip())

    for text in extra_blocks or []:
        parts.append(text.strip())

    parts.append(guidance_footer_template().format(workflow_id=workflow_id, node_id=node_id, model=model))

    return "\n\n".join(part for part in parts if part)


def resolve_templates(text: str, state: dict[str, Any], *, keep_missing: bool = True) -> str:
    """Resolve `{key}` placeholders against a state mapping.

    Used where this scaffold renders text itself (human node messages, prompt
    previews). LLM agents let ADK resolve placeholders at run time, which keeps the
    decision visible to the model without any code path.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group("key")
        value: Any = state
        if key in state:
            value = state[key]
        else:
            for part in key.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
        if value is None:
            return match.group(0) if keep_missing else ""
        if isinstance(value, (dict, list)):
            import json

            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _TEMPLATE_RE.sub(replace, text)


def describe_node_prompt(node_id: str, instruction: str, workflow_id: str, model: str, guidance: Any, tools_offered: Any) -> dict[str, Any]:
    """Inspection helper: what a node's prompt will contain, and which keys it reads."""
    assembled = assemble_instruction(
        node_id=node_id,
        instruction=instruction,
        workflow_id=workflow_id,
        model=model,
        guidance=list(guidance or []),
        tools_offered=list(tools_offered or []),
    )
    return {
        "node": node_id,
        "instruction_chars": len(assembled),
        "placeholders": placeholders(assembled),
        "guidance": list(guidance or []),
        "tools": list(tools_offered or []),
        "instruction": assembled,
    }
