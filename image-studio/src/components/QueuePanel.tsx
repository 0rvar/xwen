import { useEffect, useState } from "react";
import type { QueueItem } from "../useRenderQueue";

interface Props {
  items: QueueItem[];
  paused: boolean;
  running: boolean;
  pending: number;
  preparing: number;
  discarding: boolean;
  onStop(): void;
  onResume(): void;
  onRetry(key: string): void;
  onClear(): void;
  onDiscard(): void;
}

export function QueuePanel({ items, paused, running, pending, preparing, discarding, onStop, onResume, onRetry, onClear, onDiscard }: Props) {
  const [, setClock] = useState(0);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setClock((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  const done = items.filter((item) => item.status === "done").length;
  const failed = items.filter((item) => item.status === "error").length;
  return (
    <section className="queue-panel" aria-labelledby="queue-title">
      <header className="panel-heading queue-heading">
        <div><p className="eyebrow">Renderer</p><h2 id="queue-title">Queue</h2></div>
        <div className="queue-counts" aria-live="polite"><span>{running ? "1 running" : "Idle"}</span>{preparing > 0 && <span>{preparing} preparing</span>}<span>{pending} waiting</span><span>{done} done</span>{failed > 0 && <span className="danger-text">{failed} failed</span>}</div>
      </header>
      <div className="queue-actions">
        {paused ? <button className="quiet-button" disabled={discarding} onClick={onResume}>Resume queue</button> : <button className="quiet-button" disabled={!running && !pending && !preparing} onClick={onStop}>{running ? "Stop after current" : "Pause queue"}</button>}
        {paused && pending > 0 && <button className="text-button danger-text" disabled={preparing > 0 || discarding} onClick={onDiscard}>{discarding ? "Discarding…" : `Discard ${pending} waiting`}</button>}
        <button className="text-button" disabled={running || pending > 0 || preparing > 0 || discarding || !items.length} onClick={onClear}>Clear finished</button>
      </div>
      {!items.length ? <p className="empty-note">{preparing > 0 ? "Saving batch manifest before rendering…" : "Rendered jobs appear here. Requests run one at a time."}</p> : (
        <ol className="queue-list">
          {items.map((item) => <li key={item.key} className={`queue-item ${item.status}`}>
            <span className="status-dot" aria-hidden="true" />
            <div><strong>{item.job.request.prompt}</strong><small>Seed {item.job.request.seed} · {item.job.request.width}×{item.job.request.height}</small>{item.error && <small className="danger-text">{item.error}</small>}</div>
            <span className="queue-status">{item.status === "running" && item.startedAt ? `running ${Math.floor((Date.now() - item.startedAt) / 1000)}s` : item.status}</span>
            {item.status === "error" && <button className="text-button" disabled={discarding} onClick={() => onRetry(item.key)}>Retry</button>}
          </li>)}
        </ol>
      )}
    </section>
  );
}
