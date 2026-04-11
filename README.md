https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/en/s01-the-agent-loop.md

proxy
http://127.0.0.1:10887

s2:
Read the file requirements.txt
Create a file called greet.py with a greet(name) function
Edit greet.py to add a docstring to the function
Read greet.py to verify the edit worked

Backend (src/agents/s04_subagent.py)
FastAPI app app (only if fastapi / pydantic import succeeds):
GET /api/health — { ok, workspace }
POST /api/chat — body { "message": string, "session_id"?: string } → { session_id, events, messages, plan }
DELETE /api/sessions/{session_id} — drop server-side history
Sessions — in-memory _api_sessions (per-process).
Agent refactor (CLI + API):
LoopState.todo — per-run TodoManager (replaces global TODO).
build_tool_handlers(todo) / execute_tool_calls(..., handlers, events=...) — tool dispatch + optional event list for the UI.
_call_generation_nonstream — one-shot Generation.call (no stdout); run_one_turn(state, events=None) uses it.
CLI still prints assistant text and tool lines when events is None.
Serve:
python src/agents/s04_subagent.py serve → http://127.0.0.1:8765
CLI:
python src/agents/s04_subagent.py (prompt s04 >>).
Deps: requirements-api.txt lists fastapi and uvicorn[standard].
Frontend (web/)
Vite + React app with proxy: dev server sends /api/* to port 8765.
UI: session hint, “New session”, plan panel, per-turn events (assistant / tool / error / warning), composer.

## Run locally
```bash
pip install -r requirements-api.txt
```
If pip fails with **ResolutionImpossible** / **no matching distribution for pydantic**, your index URL is often the cause (company mirror, stale config). Try:

```bash
pip install -r requirements-api.txt -i https://pypi.org/simple
```

```bash
python src/agents/s04_subagent.py serve
```

cd web && npm install && npm run dev
Open http://127.0.0.1:5173 (Vite proxies /api to the backend).

Note on your earlier logic (used_todo / tc.get("name"))
That path is updated to use state.todo and _tool_names_in_assistant_calls (reads function.name). Reminders append after tool messages, not at index 0.

npm install did not finish in this environment (network timeout); run cd web && npm install on your machine once.