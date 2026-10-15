"""ADK 2.x: the same chat agent as a node in a graph Workflow."""

from __future__ import annotations

import os

from google.adk import Agent, Workflow

from backend.jev import route_with_jev


decision_agent = Agent(
    name="atenxion_jev_chat",
    model=os.getenv("ATENXION_WRITER_MODEL", "gemini-2.5-flash"),
    description="Atenxion chat node using Jev for routing and safety decisions.",
    tools=[route_with_jev],
    instruction=(
        "You are an Atenxion research assistant. For every user request, call "
        "route_with_jev exactly once before answering. Briefly explain the selected "
        "route, confidence, risk, and whether human review is required. Never claim "
        "that an external action was completed."
    ),
)

# A deliberately tiny ADK 2 graph. Add another node/edge only when a scenario
# actually needs it; the Jev tool and product policy remain shared in jev.py.
root_agent = Workflow(
    name="atenxion_jev_adk2",
    description="Minimal ADK 2 graph containing the Atenxion Jev chat node.",
    edges=[("START", decision_agent)],
)

