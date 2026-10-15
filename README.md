# Jev + Atenxion research

A small playground for understanding where Jev/System One decisions can help
Atenxion. It uses Google ADK's built-in development UI, so there is no custom
frontend to maintain.

```text
backend/
  jev.py       # Jev questions, mock mode, and Atenxion thresholds
  adk1.py      # ADK 1.x chat agent
  adk2.py      # ADK 2.x one-node graph workflow
agents/
  jev_demo/    # tiny adapter that lets `adk web` discover the selected runtime
tests/
.cursor/mcp.json  # official ADK documentation MCP server
```

Read `backend/jev.py` first, then `backend/adk1.py` or `backend/adk2.py`.
That is the whole experiment.

## Start the built-in chat UI

Copy the environment file and add a `GOOGLE_API_KEY` for the chat model:

```bash
cp .env.example .env
set -a; source .env; set +a
```

Jev starts in `mock` mode, so a TypeSafe key is not needed for the first run.

### ADK 1.x

```bash
uv sync --extra dev --extra adk1
ADK_RUNTIME=adk1 uv run adk web agents
```

### ADK 2.x

```bash
uv sync --extra dev --extra adk2
ADK_RUNTIME=adk2 uv run adk web agents
```

Open [http://localhost:8000](http://localhost:8000), select `jev_demo`, and
chat normally. Try these requests:

- `What is our annual leave policy?`
- `Summarize the attached contract.`
- `Refund the customer's last payment.`

The UI shows chat messages, sessions, state, events, traces, and the visual
workflow builder. It is a development UI, not a production Atenxion frontend.

ADK 1.x and ADK 2.x are separate optional environments. Running the other
`uv sync` command switches the local `.venv` to that version.

## Use live Jev

Change these values in `.env` and reload them:

```bash
JEV_MODE=live
TYPESAFE_API_KEY=your-key
TYPESAFE_DEFAULT_MODEL=jev-latest
```

The same three questions—route, risk, and human review—are sent together in one
System One request. The chat model then explains the structured result.

## Tests

```bash
uv run pytest tests/test_jev.py -q
```

Runtime-specific checks:

```bash
# In the ADK 1.x environment
uv run pytest tests/test_adk1.py -q

# In the ADK 2.x environment
uv run pytest tests/test_adk2.py -q
```

## ADK documentation MCP

`.cursor/mcp.json` configures two documentation MCP servers. Both run
`mcpdoc` through `uvx`, expose `list_doc_sources` and `fetch_docs`, and are
also installed in Hermes as `adk1-docs` and `adk2-docs`.

| Server | Source | What it serves |
| --- | --- | --- |
| `adk1-docs` | `/home/kevin/.hermes/docs/adk1-llms.txt` | Frozen ADK 1.x docs — 160 pages pinned to `google/adk-docs@94f933b73e72`, the last commit before the ADK 2.0 docs section existed |
| `adk2-docs` | `https://adk.dev/llms.txt` | Live ADK 2.x docs — Google's official index, including graphs, routes, human input, and dynamic workflows |

Rebuild the ADK 1.x index with
`python3 ~/.hermes/docs/build_adk1_index.py <sha> ~/.hermes/docs/adk1-llms.txt`
(followed by `python3 ~/.hermes/docs/validate_index.py <index> --drop`) when a
different 1.x snapshot is wanted. `mcpdoc` matches allowed domains with
`url.startswith(...)`, so the allow-list entry needs the scheme:
`https://raw.githubusercontent.com`.

Official references:

- [ADK coding with AI and MCP setup](https://adk.dev/tutorials/coding-with-ai/)
- [ADK built-in web interface](https://adk.dev/runtime/web-interface/)
- [ADK 2.0 overview and migration notes](https://adk.dev/2.0/)

## Product idea being tested

Jev does not replace Atenxion's chat model. It makes a small typed decision
before the chat model or a tool acts:

1. Which capability should handle the request?
2. How risky would an incorrect automatic action be?
3. Should a person review it first?

The thresholds stay in ordinary Python code. If the route is uncertain, risk is
high, or human-review probability is high, the result becomes `human_review`.
That pattern can later be tested around Atenxion RAG, database, document, and
action-tool features without moving this research scaffold into production yet.
