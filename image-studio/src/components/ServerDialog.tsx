import { useEffect, useState } from "react";
import type { Config } from "../domain";
import { LogTools } from "./LogTools";
import { reportError } from "../logging";

interface Props {
  config: Config;
  configPath: string;
  required: boolean;
  disabled: boolean;
  onClose(): void;
  onSave(config: Config): Promise<void>;
  onCheck(config: Config): Promise<void>;
}

export function ServerDialog({ config, configPath, required, disabled, onClose, onSave, onCheck }: Props) {
  const [draft, setDraft] = useState(config);
  const [checking, setChecking] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  useEffect(() => setDraft(config), [config]);
  const check = async () => {
    setChecking(true); setStatus(""); setError("");
    try { await onCheck(draft); setStatus("Server is reachable."); }
    catch (reason) { reportError("frontend.server.check", reason); setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setChecking(false); }
  };
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal settings-modal" role="dialog" aria-modal="true" aria-labelledby="server-title">
        <header className="modal-heading">
          <div><p className="eyebrow">Connection</p><h2 id="server-title">{required ? "Connect to xwen" : "Server settings"}</h2></div>
          {!required && <button className="icon-button" onClick={onClose} aria-label="Close settings">×</button>}
        </header>
        <p className="muted">Image Studio sends renders through your running xwen server. The URL may include a path prefix.</p>
        <label className="field"><span>Server URL</span><input autoFocus name="server-url" placeholder="http://127.0.0.1:5241" value={draft.server_url} onChange={(event) => setDraft({ ...draft, server_url: event.target.value })} /></label>
        <label className="field"><span>API key <small>optional</small></span><input name="api-key" type="password" autoComplete="off" value={draft.api_key} onChange={(event) => setDraft({ ...draft, api_key: event.target.value })} /></label>
        <p className="field-help">Saved with private file permissions at {configPath || "~/.config/xwen/image-studio.json"}.</p>
        <LogTools />
        {status && <p className="success-banner" role="status">{status}</p>}
        {error && <p className="error-banner" role="alert">{error}</p>}
        <footer className="modal-actions spread">
          <button className="quiet-button" disabled={checking || disabled || !draft.server_url.trim()} onClick={() => void check()}>{checking ? "Checking…" : "Check server"}</button>
          <button className="primary-button" disabled={checking || disabled || !draft.server_url.trim()} onClick={() => void onSave(draft).catch((reason: unknown) => { reportError("frontend.server.save", reason); setError(String(reason)); })}>Save connection</button>
        </footer>
      </section>
    </div>
  );
}
