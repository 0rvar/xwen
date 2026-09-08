import type { SavedImage } from "../domain";
import type { SessionSummary } from "../bridge";

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
  onRestore(image: SavedImage): void;
  onReveal(image: SavedImage): void;
}

function value(record: Record<string, unknown>, key: string): string | null {
  const item = record[key];
  return typeof item === "string" || typeof item === "number" ? String(item) : null;
}

export function Gallery({ images, selected, fullUrl, currentSessionId, sessions: summaries, deleting, sessionDeletionDisabled, onDelete, onDeleteSession, onSelect, onUseSource, onRestore, onReveal }: Props) {
  const sessions = new Map<string, SavedImage[]>();
  for (const image of images) sessions.set(image.session_id, [...(sessions.get(image.session_id) ?? []), image]);
  return (
    <section className="gallery-panel" aria-labelledby="gallery-title">
      <header className="panel-heading"><div><p className="eyebrow">Workspace history</p><h2 id="gallery-title">Gallery</h2></div><span className="summary-note">{images.length} image{images.length === 1 ? "" : "s"}</span></header>
      <details className="session-manager"><summary>Manage sessions</summary><div>{summaries.map((session) => <div className="session-manager-row" key={session.session_id}><span><strong>{session.session_id === currentSessionId ? "Current session" : session.session_id}</strong><small>{session.session_id} · {session.image_count} images</small></span><button className="text-button danger-button" disabled={sessionDeletionDisabled || deleting} onClick={() => onDeleteSession(session)} aria-label={`Delete session ${session.session_id}`}>Delete session</button></div>)}</div>{sessionDeletionDisabled && <p className="field-help">Finish or discard queued jobs before deleting a session.</p>}</details>
      {!images.length ? <div className="empty-gallery"><div className="empty-frame" aria-hidden="true">◇</div><h3>Your renders will collect here</h3><p>Build a prompt, preview the batch, then add it to the queue.</p></div> : (
        <div className="session-groups">
          {[...sessions].map(([sessionId, sessionImages]) => <section key={sessionId} className="session-group">
            <h3>{sessionId === currentSessionId ? "Current session" : sessionId}<span>{sessionImages.length}</span></h3>
            <div className="gallery-grid">{sessionImages.map((image) => {
              const axes = (image.metadata.context as { axes?: Record<string, unknown> } | undefined)?.axes;
              return <button key={image.id} className={`gallery-tile ${selected?.id === image.id ? "selected" : ""}`} onClick={() => onSelect(image)} aria-label={`Open ${image.prompt}, seed ${image.seed}`}>
                <img src={image.data_url} alt="" />
                <span className="gallery-caption"><strong>{image.prompt}</strong><small>Seed {image.seed}{axes && Object.keys(axes).length ? ` · ${Object.values(axes).join(" · ")}` : ""}</small></span>
              </button>;
            })}</div>
          </section>)}
        </div>
      )}
      {selected && <div className="modal-backdrop preview-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onSelect(selected); }}>
        <section className="modal preview-modal" role="dialog" aria-modal="true" aria-label="Image preview">
          <div className="preview-image"><img src={fullUrl || selected.data_url} alt={selected.prompt} /></div>
          <aside className="preview-info">
            <button className="icon-button preview-close" onClick={() => onSelect(selected)} aria-label="Close preview">×</button>
            <p className="eyebrow">Generated image</p><h2>{selected.prompt}</h2>
            <dl><div><dt>Seed</dt><dd>{selected.seed}</dd></div><div><dt>Size</dt><dd>{selected.width}×{selected.height}</dd></div>{value(selected.metadata, "created_at") && <div><dt>Created</dt><dd>{value(selected.metadata, "created_at")}</dd></div>}{value(selected.metadata, "duration_ms") && <div><dt>Duration</dt><dd>{(Number(value(selected.metadata, "duration_ms")) / 1000).toFixed(1)}s</dd></div>}</dl>
            <details className="metadata-details"><summary>Saved metadata</summary><pre>{JSON.stringify(selected.metadata, null, 2)}</pre></details>
            <div className="preview-actions"><button className="primary-button" onClick={() => onUseSource(selected)}>Use as source</button><button className="quiet-button" onClick={() => onRestore(selected)}>Restore settings</button><button className="text-button" onClick={() => onReveal(selected)}>Show in Finder</button><button className="text-button danger-button" disabled={deleting} onClick={() => onDelete(selected)}>Delete image</button></div>
          </aside>
        </section>
      </div>}
    </section>
  );
}
