"""Tiny bridge used only by the built-in `adk web` development UI."""

import os

if os.getenv("ADK_RUNTIME", "adk1").lower() == "adk2":
    from backend.adk2 import root_agent
else:
    from backend.adk1 import root_agent

__all__ = ["root_agent"]

