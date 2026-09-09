import { useRef, useState } from "react";
import { bridge } from "../bridge";
import { reportError } from "../logging";

export function PromptGenerator({ prompt, disabled, onUse }: { prompt: string; disabled: boolean; onUse(prompt: string): void }) {
  const [open, setOpen] = useState(false);
  const [idea, setIdea] = useState("");
  const [draft, setDraft] = useState("");
  const [hasDraft, setHasDraft] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestToken = useRef(0);
  const close = () => { requestToken.current += 1; setBusy(false); setDraft(""); setOpen(false); };
  const generate = async () => {
    const token = ++requestToken.current;
    setBusy(true); setError(""); setDraft(""); setHasDraft(false);
    try { const result = await bridge.generatePrompt(idea); if (requestToken.current === token) { setDraft(result); setHasDraft(true); } }
    catch (reason) { reportError("frontend.prompt", reason); if (requestToken.current === token) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (requestToken.current === token) setBusy(false); }
  };
  return <>
    <button className="text-button" disabled={disabled} onClick={() => { setIdea(prompt); setDraft(""); setHasDraft(false); setError(""); setOpen(true); }}>Draft image prompt</button>
    {open && <div className="modal-backdrop" role="presentation"><section className="modal prompt-generator" role="dialog" aria-modal="true" aria-labelledby="prompt-generator-title">
      <p className="eyebrow">xwen</p><h2 id="prompt-generator-title">Prompt generator</h2>
      <p className="muted">Describe an idea, or leave it blank for a surprise. Uses your configured xwen server.</p>
      <label className="field"><span>Idea or instructions</span><textarea rows={4} value={idea} disabled={busy} onChange={(event) => setIdea(event.target.value)} /></label>
      <button className="quiet-button" disabled={busy || disabled} onClick={() => void generate()}>{busy ? "Drafting…" : "Create draft"}</button>
      {error && <p className="error-banner" role="alert">{error}</p>}
      {hasDraft && <label className="field"><span>Draft prompt</span><textarea rows={7} value={draft} onChange={(event) => setDraft(event.target.value)} /></label>}
      <div className="dialog-actions"><button className="quiet-button" onClick={close}>Close generator</button><button className="primary-button" disabled={busy || !draft.trim()} onClick={() => { onUse(draft.trim()); close(); }}>Use prompt</button></div>
    </section></div>}
  </>;
}
