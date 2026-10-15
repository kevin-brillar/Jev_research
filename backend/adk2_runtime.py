"""ADK 2.8 runtime: a spec becomes a graph Workflow.

Only this module knows ADK 2 node types, and it owns exactly three translations:

* `llm`      -> `Agent(..., mode="single_turn")`, whose instruction is the assembled
                prompt from config; `output_key` writes its answer into state
* `function` -> `FunctionNode` wrapping a registry function; the node's returned
                route becomes `Event(route=...)` and its state updates are applied
* `jev`      -> `FunctionNode` running the shared Jev-node semantics, so the
                decision is a *route the graph follows*, not advice a model may skip
* `human`    -> `FunctionNode` that yields `RequestInput`, so the run pauses until a
                person answers; on resume the reply selects the route
* `router`   -> `FunctionNode` evaluating the spec's conditions over state

Termination nodes are dropped: in a graph the branch simply ends.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from google.adk import Agent, Event, Workflow
from google.adk.events import RequestInput

from backend import functions as node_functions
from backend import spec as spec_module
from backend import tools as tool_registry
from backend.compiler import (
    BuildContext,
    Graph,
    NodeOutcome,
    apply_state,
    compile_workflow,
    run_jev_node,
    run_router_node,
)

APPROVE_WORDS = ("approve", "approved", "yes", "accept")


def _model_for(ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
    """Resolve a node's model; the spec may name `stub` for offline runs."""
    name = ctx.model_for(node)
    if name == "stub":
        from backend.stub_llm import StubLlm

        return StubLlm(model=f"stub:{node.id}")
    return name


def _reply_text(node_input: Any) -> str:
    if isinstance(node_input, str):
        return node_input
    if isinstance(node_input, dict):
        for key in ("text", "response", "result", "user_response", "reply"):
            if isinstance(node_input.get(key), str):
                return node_input[key]
        return str(node_input)
    return str(node_input or "")


class Adk2Runtime:
    """Runtime adapter for google-adk 2.8 (atenxion-agentic-gemini-2.0's lane)."""

    name = "adk2"

    # ------------------------------------------------------------------ nodes
    def build_llm(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        return Agent(
            name=node.id,
            description=node.description or node.label,
            model=_model_for(ctx, node),
            instruction=ctx.instruction_for(node),
            tools=tool_registry.callables(node.tools),
            mode="single_turn",  # ADK requires graph agents to be single-turn/task
            output_key=node.output_key,
        )

    def build_function(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        resolved = node_functions.resolve_function(node.function or "", **(node.params or {}))

        async def node_callable(context: Any, node_input: Any) -> Event:
            outcome = await resolved(dict(context.state), node_input)
            apply_state(context.state, outcome.state)
            _append_log(context.state, node.id, outcome.log)
            return Event(route=outcome.route, output=outcome.output)

        node_callable.__name__ = node.id
        return _function_node(node_callable, node.id)

    def build_jev(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        async def node_callable(context: Any, node_input: Any) -> Event:
            outcome = await run_jev_node(node, context.state, ctx)
            apply_state(context.state, outcome.state)
            _append_log(context.state, node.id, outcome.log)
            return Event(route=outcome.route, output=outcome.output)

        node_callable.__name__ = node.id
        return _function_node(node_callable, node.id)

    def build_router(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        async def node_callable(context: Any, node_input: Any) -> Event:
            outcome = await run_router_node(node, context.state)
            apply_state(context.state, outcome.state)
            _append_log(context.state, node.id, outcome.log)
            return Event(route=outcome.route, output=outcome.output)

        node_callable.__name__ = node.id
        return _function_node(node_callable, node.id)

    def build_human(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        approve_words = tuple(node.params.get("approve_words", APPROVE_WORDS))
        reject_words = tuple(node.params.get("reject_words", ("reject", "rejected", "no", "decline")))

        async def node_callable(context: Any, node_input: Any):
            """Pause for a person; the reply selects the route."""
            prompt = node.message
            for key, value in dict(context.state).items():
                prompt = prompt.replace("{" + key + "}", str(value))
            interrupt_id = _interrupt_id(node.id)
            resume_inputs = getattr(context, "resume_inputs", {}) or {}
            reply = resume_inputs.get(interrupt_id)
            if reply is None:
                yield RequestInput(
                    interrupt_id=interrupt_id,
                    message=prompt,
                    payload={"node": node.id, "pending_action": context.state.get("case.pending_action")},
                    response_schema=str,
                )
                return
            text = _reply_text(reply).lower()
            approved = any(word in text for word in approve_words)
            route = node.routes.get("approved" if approved else "rejected") or node.default_route
            apply_state(
                context.state,
                {
                    f"approvals.{node.id}": {"approved": approved, "reply": _reply_text(reply)},
                    f"case.events": [*(context.state.get("case.events") or []), {"node": node.id, "approved": approved}],
                },
            )
            _append_log(context.state, node.id, [f"{node.id}: {'approved' if approved else 'rejected'} -> {route}"])
            yield Event(route=route, output={"approved": approved, "reply": _reply_text(reply)})

        node_callable.__name__ = node.id
        # rerun_on_resume: the body must re-run to choose the route from the reply.
        return _function_node(node_callable, node.id, rerun_on_resume=True)

    def build_termination(self, ctx: BuildContext, node: spec_module.NodeSpec) -> Any:
        return None

    # ---------------------------------------------------------------- assembly
    def finalize(self, ctx: BuildContext, graph: Graph) -> Any:
        edges: list[Any] = [("START", graph.nodes[ctx.spec.entry])]

        for node_id, node in ctx.spec.nodes.items():
            built = graph.nodes.get(node_id)
            if built is None:  # termination
                continue
            routes = graph.route_map.get(node_id) or {}
            if routes:
                mapping = {route: graph.nodes[target] for route, target in routes.items() if target in graph.nodes}
                default = graph.default_routes.get(node_id)
                if default and default in graph.nodes:
                    from google.adk.workflow import DEFAULT_ROUTE

                    mapping[DEFAULT_ROUTE] = graph.nodes[default]
                if mapping:
                    edges.append((built, mapping))
            for source, target in graph.structural_edges:
                if source == node_id and target in graph.nodes and graph.nodes[target] is not None:
                    edges.append((built, graph.nodes[target]))

        return Workflow(
            name=ctx.spec.id,
            description=ctx.spec.description or ctx.spec.name,
            edges=edges,
        )


def _function_node(callable_: Any, name: str, *, rerun_on_resume: bool = False) -> Any:
    from google.adk.workflow import FunctionNode

    return FunctionNode(func=callable_, name=name, rerun_on_resume=rerun_on_resume)


def _interrupt_id(node_id: str) -> str:
    """Stable interrupt id: a random one per call would break resume matching."""
    return f"human:{node_id}"


def _append_log(state: MutableMapping[str, Any], node_id: str, lines: list[str]) -> None:
    if not lines:
        return
    state["run.log"] = [*(state.get("run.log") or []), *lines]


def build(
    name: str,
    *,
    ledger: Any = None,
    model_override: str | None = None,
) -> Any:
    """Compile a workflow spec into an ADK 2 root agent."""
    return compile_workflow(name, Adk2Runtime(), ledger=ledger, model_override=model_override).root
