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
  const orderedItems = [...items].sort((a, b) => {
    const rank = { running: 0, pending: 1, error: 2, done: 3 } as const;
    return rank[a.status] - rank[b.status];
  });
  const activeItems = orderedItems.filter((item) => item.status !== "done");
  const completedItems = orderedItems.filter((item) => item.status === "done");
  const renderItem = (item: QueueItem) => <li key={item.key} className={`queue-item ${item.status}`}>
    <span className="status-dot" aria-hidden="true" />
    <div><strong>{item.job.request.prompt}</strong><small>Seed {item.job.request.seed} · {item.job.request.width}×{item.job.request.height} · {item.job.request.steps} steps · {item.job.context.mode}</small>{item.job.request.loras.length > 0 && <small title={item.job.request.loras.map((lora) => `${lora.name} @ ${lora.weight}`).join(" · ")}>LoRAs: {item.job.request.loras.map((lora) => `${lora.name.split(/[\\/]/).pop()} @ ${lora.weight}`).join(" · ")}</small>}{item.error && <small className="danger-text">{item.error}</small>}{item.status === "running" && <progress className="queue-progress" aria-label="Rendering" />}</div>
    <span className="queue-status">{item.status === "running" && item.startedAt ? `running ${Math.floor((Date.now() - item.startedAt) / 1000)}s` : item.status}</span>
    {item.status === "error" && <button className="text-button" disabled={discarding} onClick={() => onRetry(item.key)}>Retry</button>}
  </li>;
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
        <div className="queue-sections">
          {activeItems.length > 0 ? <ol className="queue-list">{activeItems.map(renderItem)}</ol> : <p className="queue-empty-active">No queued or running jobs.</p>}
          {completedItems.length > 0 && <section className="queue-completed" aria-label="Completed jobs"><h3>Completed</h3><ol className="queue-list">{completedItems.map(renderItem)}</ol></section>}
        </div>
      )}
    </section>
  );
}
