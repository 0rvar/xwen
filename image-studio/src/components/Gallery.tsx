import { useState } from "react";
import type { SavedImage } from "../domain";
import type { SessionSummary } from "../bridge";
import { GenerationDetails } from "./GenerationDetails";

interface Props {
  images: SavedImage[];
  selected: SavedImage | null;
  fullUrl: string;
  currentSessionId: string | null;
  sessions: SessionSummary[];
  deleting: boolean;
  sessionDeletionDisabled: boolean;
  onDelete(image: SavedImage): void;
  onDeleteSession(session: SessionSummary): void;
  onSelect(image: SavedImage): void;
  onUseSource(image: SavedImage): void;
  onRestore(image: SavedImage, fields?: string[]): void;
  onReveal(image: SavedImage): void;
}

function value(record: Record<string, unknown>, key: string): string | null {
  const item = record[key];
  return typeof item === "string" || typeof item === "number" ? String(item) : null;
}

export function Gallery({ images, selected, fullUrl, currentSessionId, sessions: summaries, deleting, sessionDeletionDisabled, onDelete, onDeleteSession, onSelect, onUseSource, onRestore, onReveal }: Props) {
  const [restoreField, setRestoreField] = useState("all");
  const [comparison, setComparison] = useState<SavedImage[] | null>(null);
  const sessions = new Map<string, SavedImage[]>();
  for (const image of images) sessions.set(image.session_id, [...(sessions.get(image.session_id) ?? []), image]);
  const sessionRows = new Map(summaries.map((session) => [session.session_id, session]));
  for (const [session_id, sessionImages] of sessions) {
    if (!sessionRows.has(session_id)) sessionRows.set(session_id, { session_id, image_count: sessionImages.length });
  }
  return (
    <section className="gallery-panel" aria-labelledby="gallery-title">
      <header className="panel-heading"><div><p className="eyebrow">Workspace history</p><h2 id="gallery-title">Gallery</h2></div><span className="summary-note">{images.length} image{images.length === 1 ? "" : "s"}</span></header>
        <div className="session-groups">
          {[...sessionRows.values()].map((session) => <section key={session.session_id} className="session-group">
            <div className="session-heading"><h3>{session.session_id}{session.session_id === currentSessionId ? " · Current session" : ""}<span>{session.image_count}</span></h3><button className="text-button danger-button" disabled={sessionDeletionDisabled || deleting} title={sessionDeletionDisabled ? "Finish or discard queued jobs before deleting a session" : undefined} onClick={() => onDeleteSession(session)} aria-label={`Delete session ${session.session_id}`}>Delete session</button></div>
            {[...new Map((sessions.get(session.session_id) ?? []).map((image) => [String((image.metadata.context as Record<string, unknown> | undefined)?.batch_id ?? ""), image])).keys()].filter(Boolean).map((batchId) => {
              const batch = (sessions.get(session.session_id) ?? []).filter((image) => (image.metadata.context as Record<string, unknown> | undefined)?.batch_id === batchId);
              const axes = Object.keys(((batch[0]?.metadata.context as Record<string, unknown> | undefined)?.axes as Record<string, unknown> | undefined) ?? {});
              return axes.length > 0 && axes.length <= 2 && batch.length > 1 ? <button key={batchId} className="comparison-button" onClick={() => setComparison(batch)}>Compare batch · {axes.join(" × ")} ({batch.length})</button> : null;
            })}
            <div className="gallery-grid">{(sessions.get(session.session_id) ?? []).map((image) => {
              const axes = (image.metadata.context as { axes?: Record<string, unknown> } | undefined)?.axes;
              return <button key={image.id} className={`gallery-tile ${selected?.id === image.id ? "selected" : ""}`} onClick={() => onSelect(image)} aria-label={`Open ${image.prompt}, seed ${image.seed}`}>
                <img src={image.data_url} alt="" />
                <span className="gallery-caption"><strong>{image.prompt}</strong><small>Seed {image.seed}{axes && Object.keys(axes).length ? ` · ${Object.values(axes).join(" · ")}` : ""}</small></span>
              </button>;
            })}</div>
          </section>)}
        </div>
      {!images.length && <div className="empty-gallery"><div className="empty-frame" aria-hidden="true">◇</div><h3>Your renders will collect here</h3><p>Build a prompt, preview the batch, then add it to the queue.</p></div>}
      {selected && <div className="modal-backdrop preview-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onSelect(selected); }}>
        <section className="modal preview-modal" role="dialog" aria-modal="true" aria-label="Image preview">
          <div className="preview-image"><img src={fullUrl || selected.data_url} alt={selected.prompt} /></div>
          <aside className="preview-info">
            <button className="icon-button preview-close" onClick={() => onSelect(selected)} aria-label="Close preview">×</button>
            <p className="eyebrow">Generated image</p><h2>{selected.prompt}</h2>
            <dl><div><dt>Seed</dt><dd>{selected.seed}</dd></div><div><dt>Size</dt><dd>{selected.width}×{selected.height}</dd></div>{value(selected.metadata, "created_at") && <div><dt>Created</dt><dd>{value(selected.metadata, "created_at")}</dd></div>}{value(selected.metadata, "duration_ms") && <div><dt>Duration</dt><dd>{(Number(value(selected.metadata, "duration_ms")) / 1000).toFixed(1)}s</dd></div>}</dl>
            <GenerationDetails metadata={selected.metadata} />
            <details className="metadata-details"><summary>Saved metadata</summary><pre>{JSON.stringify(selected.metadata, null, 2)}</pre></details>
            <div className="preview-actions"><button className="primary-button" onClick={() => onUseSource(selected)}>Use as source</button><div className="restore-row"><select aria-label="Settings to restore" value={restoreField} onChange={(event) => setRestoreField(event.target.value)}><option value="all">All settings</option><option value="prompt">Prompt</option><option value="dimensions">Dimensions</option><option value="steps">Steps</option><option value="seed">Seed</option><option value="loras">LoRAs</option><option value="edit">Edit inputs</option></select><button className="quiet-button" onClick={() => { onRestore(selected, restoreField === "all" ? undefined : [restoreField]); onSelect(selected); }}>Restore selected</button></div><button className="text-button" onClick={() => onReveal(selected)}>Show in Finder</button><button className="text-button danger-button" disabled={deleting} onClick={() => onDelete(selected)}>Delete image</button></div>
          </aside>
        </section>
      </div>}
      {comparison && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) setComparison(null); }}><section className="modal comparison-modal" role="dialog" aria-modal="true" aria-label="Batch comparison"><div className="modal-heading"><div><p className="eyebrow">Batch comparison</p><h2>{comparison.length} rendered variations</h2></div><button className="icon-button" onClick={() => setComparison(null)} aria-label="Close comparison">×</button></div><ComparisonGrid images={comparison} onSelect={(image) => { setComparison(null); onSelect(image); }} /></section></div>}
    </section>
  );
}

function ComparisonGrid({ images, onSelect }: { images: SavedImage[]; onSelect(image: SavedImage): void }) {
  const axes = Object.keys(((images[0]?.metadata.context as Record<string, unknown> | undefined)?.axes as Record<string, unknown> | undefined) ?? {});
  const values = axes.map((axis) => [...new Set(images.map((image) => String(((image.metadata.context as Record<string, unknown> | undefined)?.axes as Record<string, unknown> | undefined)?.[axis] ?? ""))) ]);
  if (axes.length === 1) return <div className="comparison-grid one-dimensional">{images.map((image) => <button key={image.id} className="comparison-tile" onClick={() => onSelect(image)}><strong>{String(((image.metadata.context as Record<string, unknown> | undefined)?.axes as Record<string, unknown> | undefined)?.[axes[0]!] ?? "")}</strong><img src={image.data_url} alt="" /></button>)}</div>;
  const columnAxis = values[0]!.length <= values[1]!.length ? 0 : 1;
  const rowAxis = columnAxis === 0 ? 1 : 0;
  const imageAt = (row: string, column: string) => images.find((image) => { const imageAxes = ((image.metadata.context as Record<string, unknown> | undefined)?.axes as Record<string, unknown> | undefined) ?? {}; return String(imageAxes[axes[rowAxis]!] ?? "") === row && String(imageAxes[axes[columnAxis]!] ?? "") === column; });
  return <div className="comparison-table" style={{ "--comparison-columns": values[columnAxis]!.length } as React.CSSProperties}><div className="comparison-axis-label">{axes[rowAxis]} ↓ / {axes[columnAxis]} →</div>{values[columnAxis]!.map((value) => <strong key={value} className="comparison-column-label">{value}</strong>)}{values[rowAxis]!.flatMap((row) => [<strong key={`${row}-label`} className="comparison-row-label">{row}</strong>, ...values[columnAxis]!.map((column) => { const image = imageAt(row, column); return image ? <button key={image.id} className="comparison-tile" onClick={() => onSelect(image)}><img src={image.data_url} alt="" /></button> : <span key={`${row}-${column}`} />; })])}</div>;
}
