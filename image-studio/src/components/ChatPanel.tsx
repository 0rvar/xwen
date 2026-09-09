import { useEffect, useRef, useState } from "react";
import { bridge } from "../bridge";
import { captureChatDefaults, runChatTurn, type ChatMessage } from "../chatTurn";
import type { LoraCandidate, RenderRequest, StudioSettings } from "../domain";
import { reportError } from "../logging";

export function ChatPanel({ open, onClose, settings, loras, sessionId, disabled, onQueue, acquireTurn }: {
  open: boolean; onClose(): void; settings: StudioSettings; loras: LoraCandidate[];
  sessionId: string | null; disabled: boolean; onQueue(requests: RenderRequest[]): Promise<void>;
  acquireTurn(): Promise<() => void>;
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState("");
  const [staged, setStaged] = useState(0);
  const [queued, setQueued] = useState(0);
  const historyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const history = historyRef.current;
    if (history) history.scrollTop = history.scrollHeight;
  }, [messages, staged, busy]);

  const send = async () => {
    if (!input.trim() || busyRef.current || disabled || !sessionId) return;
    busyRef.current = true;
    setBusy(true); setError(""); setStaged(0); setQueued(0);
    const history: ChatMessage[] = [...messages, { role: "user", content: input.trim() }];
    setMessages(history); setInput("");
    let release: (() => void) | undefined;
    try {
      const defaults = captureChatDefaults(settings);
      const availableLoras = structuredClone(loras);
      release = await acquireTurn();
      const result = await runChatTurn({ history, defaults, loras: availableLoras, chat: bridge.chat, onQueue, onStaged: setStaged });
      setMessages(result.history); setQueued(result.queued);
    } catch (reason) {
      reportError("frontend.chat", reason);
      setError(`${reason instanceof Error ? reason.message : String(reason)} No jobs were queued for this turn.`);
    } finally {
      release?.();
      busyRef.current = false;
      setBusy(false); setStaged(0);
    }
  };

  return <aside id="image-assistant" className="chat-sidebar" aria-hidden={!open} inert={!open} aria-labelledby="chat-title">
    <div className="chat-sidebar-content">
      <div className="chat-heading"><div><p className="eyebrow">xwen</p><h2 id="chat-title">Image assistant</h2></div><button className="icon-button" onClick={onClose} aria-label="Close chat">×</button></div>
      <div className="chat-history" aria-live="polite" ref={historyRef}>
        {messages.length === 0 && <p className="muted">Ask for images, variations, or a LoRA comparison. The assistant prepares the complete batch before generation starts.</p>}
        {messages.filter((message) => (message.role === "user" || message.role === "assistant") && typeof message.content === "string" && message.content.trim()).map((message, index) => <div className={`chat-message ${message.role}`} key={index}><strong>{message.role === "user" ? "You" : "Assistant"}</strong><p>{message.content}</p></div>)}
        {busy && <p className="field-help" role="status">{staged ? `${staged} ${staged === 1 ? "image" : "images"} staged. Finishing the complete batch…` : "Preparing your reply. Any current image finishes first…"}</p>}
        {!busy && queued > 0 && <p className="field-help" role="status">{queued} {queued === 1 ? "image" : "images"} queued.</p>}
        {error && <p className="error-banner" role="alert">{error}</p>}
      </div>
      <div className="chat-composer">
        <label className="field"><span>Message</span><textarea rows={3} value={input} disabled={busy || disabled} placeholder="Make a warm terracotta garden pavilion…" onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); void send(); } }} /></label>
        <div className="dialog-actions"><span className="field-help">⌘↵ to send</span><button className="quiet-button" disabled={busy} onClick={() => { if (busyRef.current) return; setMessages([]); setError(""); setQueued(0); }}>Clear</button><button className="primary-button" disabled={busy || disabled || !input.trim() || !sessionId} onClick={() => void send()}>{busy ? "Thinking…" : "Send"}</button></div>
      </div>
    </div>
  </aside>;
}
