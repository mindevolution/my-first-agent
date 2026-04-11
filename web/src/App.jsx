import { useCallback, useState } from "react";
import "./App.css";

const api = (path, opts) => fetch(path, opts);

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [plan, setPlan] = useState("");
  const [turns, setTurns] = useState([]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setLoading(true);
    setError(null);
    setTurns((t) => [...t, { role: "user", content: text }]);
    try {
      const res = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || res.statusText || "Request failed");
      }
      setSessionId(data.session_id);
      setPlan(data.plan || "");
      setTurns((t) => [
        ...t,
        {
          role: "agent",
          events: data.events || [],
          messages: data.messages,
        },
      ]);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId]);

  const newSession = useCallback(async () => {
    if (sessionId) {
      await api(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      }).catch(() => {});
    }
    setSessionId(null);
    setPlan("");
    setTurns([]);
    setError(null);
  }, [sessionId]);

  return (
    <div className="app">
      <header className="header">
        <h1>s04 subagent</h1>
        <p className="sub">
          Workspace agent UI — start the API:{" "}
          <code className="code">
            python src/agents/s04_subagent.py serve
          </code>
        </p>
        <div className="actions">
          <button type="button" className="btn secondary" onClick={newSession}>
            New session
          </button>
          {sessionId && (
            <span className="sid" title="Session id">
              session …{sessionId.slice(-8)}
            </span>
          )}
        </div>
      </header>

      {plan && (
        <section className="plan">
          <h2>Plan</h2>
          <pre>{plan}</pre>
        </section>
      )}

      <section className="log">
        <h2>Conversation</h2>
        {turns.length === 0 && (
          <p className="muted">Send a message to run the agent in your project cwd.</p>
        )}
        {turns.map((block, i) => (
          <article key={i} className={`block ${block.role}`}>
            {block.role === "user" ? (
              <pre className="user-msg">{block.content}</pre>
            ) : (
              <AgentTurn events={block.events} />
            )}
          </article>
        ))}
      </section>

      {error && <div className="err">{error}</div>}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          rows={3}
          value={input}
          placeholder="Ask the agent (e.g. create a file, run a command)…"
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="btn primary" disabled={loading}>
          {loading ? "Running…" : "Send"}
        </button>
      </form>
    </div>
  );
}

function AgentTurn({ events }) {
  if (!events?.length) {
    return <p className="muted">No events.</p>;
  }
  return (
    <ul className="events">
      {events.map((ev, j) => (
        <li key={j} className={`ev ev-${ev.type}`}>
          {ev.type === "assistant" && (
            <>
              <span className="tag">assistant</span>
              {ev.finish_reason && (
                <span className="fr">({ev.finish_reason})</span>
              )}
              <pre>{ev.content || "(tool calls only)"}</pre>
            </>
          )}
          {ev.type === "tool" && (
            <>
              <span className="tag">tool: {ev.name}</span>
              {ev.command != null && (
                <pre className="cmd">{ev.command}</pre>
              )}
              {ev.path != null && (
                <pre className="cmd">{ev.path}</pre>
              )}
              {ev.output_preview && (
                <pre className="out">{ev.output_preview}</pre>
              )}
            </>
          )}
          {ev.type === "error" && (
            <>
              <span className="tag err">error</span>
              <pre>{ev.message}</pre>
            </>
          )}
          {ev.type === "warning" && (
            <>
              <span className="tag warn">warning</span>
              <pre>{ev.message}</pre>
            </>
          )}
        </li>
      ))}
    </ul>
  );
}
