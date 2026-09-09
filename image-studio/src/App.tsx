import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bridge } from "./bridge";
import type { SessionSummary } from "./bridge";
import type { BatchAxis, BatchMode, Config, InputImage, LoraCandidate, PlannedJob, SavedImage, StudioSettings, Workspace } from "./domain";
import { DEFAULT_SETTINGS, planJobs, restoreSettings, snapSize } from "./domain";
import { useRenderQueue } from "./useRenderQueue";
import { BatchPanel } from "./components/BatchPanel";
import { Gallery } from "./components/Gallery";
import { MaskEditor } from "./components/MaskEditor";
import { QueuePanel } from "./components/QueuePanel";
import { ServerDialog } from "./components/ServerDialog";
import { SettingsPanel } from "./components/SettingsPanel";
import { useImageDrop, type ImageDropTarget } from "./useImageDrop";
import { reportError, setLogSecrets } from "./logging";

const EMPTY_CONFIG: Config = { server_url: "", api_key: "", workspaces: [], last_workspace: null };

function errorText(reason: unknown): string {
  reportError("frontend.operation", reason);
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  try { return JSON.stringify(reason); } catch { return "An unexpected error occurred."; }
}

function assetPath(image: SavedImage, relative: string): string {
  const base = image.metadata_path.replace(/[/\\][^/\\]+$/, "");
  return `${base}/${relative}`;
}

function assetReference(metadata: Record<string, unknown>, key: "init_image" | "mask" | "control.image"): string | null {
  const assets = metadata.assets;
  if (!assets || typeof assets !== "object") return null;
  const entry = (assets as Record<string, unknown>)[key];
  if (!entry || typeof entry !== "object") return null;
  const path = (entry as Record<string, unknown>).path;
  return typeof path === "string" ? path : null;
}

export default function App() {
  const [loading, setLoading] = useState(true);
  const [fatal, setFatal] = useState("");
  const [notice, setNotice] = useState("");
  const [config, setConfig] = useState<Config>(EMPTY_CONFIG);
  const [configPath, setConfigPath] = useState("");
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [settings, setSettings] = useState<StudioSettings>(DEFAULT_SETTINGS);
  const [images, setImages] = useState<SavedImage[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [deleteTarget, setDeleteTarget] = useState<{ workspacePath: string; image?: SavedImage; session?: SessionSummary } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [selected, setSelected] = useState<SavedImage | null>(null);
  const [fullUrl, setFullUrl] = useState("");
  const [serverOpen, setServerOpen] = useState(false);
  const [serverRequired, setServerRequired] = useState(false);
  const [workspacePrompt, setWorkspacePrompt] = useState(false);
  const [loras, setLoras] = useState<LoraCandidate[]>([]);
  const [loraLoading, setLoraLoading] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [controlPreview, setControlPreview] = useState<InputImage | null>(null);
  const [controlBusy, setControlBusy] = useState(false);
  const [axes, setAxes] = useState<BatchAxis[]>([]);
  const [batchMode, setBatchMode] = useState<BatchMode>("matrix");
  const [preview, setPreview] = useState<PlannedJob[]>([]);
  const [batchError, setBatchError] = useState("");
  const selectionToken = useRef(0);
  const controlToken = useRef(0);
  const historyToken = useRef(0);

  const applyImportedImage = useCallback((kind: ImageDropTarget, image: InputImage) => {
    setSettings((current) => {
      if (kind === "control") return { ...current, controlImage: image };
      const snapped = snapSize(image.width, image.height);
      return { ...current, initImage: image, width: snapped.width, height: snapped.height, mask: null };
    });
    setPreview([]); setBatchError(""); setNotice("");
    controlToken.current += 1; setControlPreview(null); setControlBusy(false);
  }, []);
  const imageDrop = useImageDrop(applyImportedImage, setNotice);

  const addOutputs = useCallback((created: SavedImage[]) => {
    setImages((current) => [...created, ...current.filter((image) => !created.some((next) => next.id === image.id))]);
    const token = ++historyToken.current;
    void bridge.listSessions().then((result) => { if (historyToken.current === token) setSessions(result); }).catch((reason: unknown) => { if (historyToken.current === token) setNotice(errorText(reason)); });
  }, []);
  const queue = useRenderQueue(addOutputs);

  const refreshImages = useCallback(async () => {
    const token = ++historyToken.current;
    const [saved, allSessions] = await Promise.all([bridge.listImages(), bridge.listSessions()]);
    if (historyToken.current === token) { setImages(saved); setSessions(allSessions); }
  }, []);
  useEffect(() => {
    let disposed = false;
    void bridge.bootstrap().then(async (result) => {
      if (disposed) return;
      setLogSecrets(result.config.api_key);
      setConfig(result.config);
      setConfigPath(result.config_path);
      setWorkspace(result.workspace);
      if (result.warning) setNotice(result.warning);
      const missingServer = !result.config.server_url.trim();
      setServerRequired(missingServer);
      setServerOpen(missingServer);
      setWorkspacePrompt(!result.workspace && !missingServer);
      if (result.workspace) await refreshImages();
    }).catch((reason: unknown) => setFatal(errorText(reason))).finally(() => setLoading(false));
    return () => { disposed = true; };
  }, [refreshImages]);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (deleteTarget) { if (!deleting) setDeleteTarget(null); }
        else if (selected) { setSelected(null); setFullUrl(""); }
        else if (maskOpen) setMaskOpen(false);
        else if (serverOpen && !serverRequired) setServerOpen(false);
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [maskOpen, selected, serverOpen, serverRequired, deleteTarget, deleting]);

  const changeSettings = (next: StudioSettings) => {
    setSettings(next);
    setPreview([]);
    setBatchError("");
    if (next.controlImage?.data_url !== settings.controlImage?.data_url || next.controlKind !== settings.controlKind || next.width !== settings.width || next.height !== settings.height) {
      controlToken.current += 1;
      setControlPreview(null);
      setControlBusy(false);
    }
  };
  const chooseWorkspace = async (path?: string, newSession = false) => {
    if (queue.active) return;
    const chosen = path ?? await bridge.chooseDirectory(newSession ? "Choose a workspace for the new session" : "Choose an Image Studio workspace");
    if (!chosen) { setWorkspacePrompt(true); return; }
    try {
      historyToken.current += 1;
      const next = await bridge.selectWorkspace(chosen);
      setWorkspace(next);
      setConfig((current) => ({ ...current, last_workspace: next.path, workspaces: [next.path, ...current.workspaces.filter((item) => item !== next.path)] }));
      setWorkspacePrompt(false); setImages([]); setSelected(null); setNotice(""); queue.clearFinished();
      await refreshImages();
    } catch (reason) { setNotice(errorText(reason)); setWorkspacePrompt(true); }
  };
  const pickImage = async (kind: "source" | "control") => {
    const path = await bridge.chooseImage(kind === "source" ? "Choose a source image" : "Choose a ControlNet image");
    if (!path) return;
    imageDrop.importPath(kind, path);
  };
  const refreshLoras = async () => {
    setLoraLoading(true);
    try { setLoras(await bridge.listLoras()); setNotice(""); }
    catch (reason) { setNotice(errorText(reason)); }
    finally { setLoraLoading(false); }
  };
  const generate = () => {
    if (!workspace) return;
    try { queue.enqueue(planJobs(settings), workspace.session_id); setNotice(""); }
    catch (reason) { setNotice(errorText(reason)); }
  };
  const chooseSelected = async (image: SavedImage) => {
    const token = ++selectionToken.current;
    if (selected?.id === image.id) { setSelected(null); setFullUrl(""); return; }
    setSelected(image); setFullUrl(image.data_url);
    try {
      const loaded = await bridge.readImage(image.path);
      if (selectionToken.current === token) setFullUrl(loaded.data_url);
    } catch { /* The thumbnail remains usable and the metadata stays accessible. */ }
  };
  const useAsSource = async (image: SavedImage) => {
    try {
      const input = await bridge.readImage(image.path);
      changeSettings({ ...settings, mode: "img2img", initImage: input, mask: null, width: image.width, height: image.height });
      setSelected(null); setFullUrl("");
    } catch (reason) { setNotice(errorText(reason)); }
  };
  const restore = async (image: SavedImage) => {
    try {
      const restored = restoreSettings(image.metadata, settings);
      const [initImage, mask, controlImage] = await Promise.all(([
        ["init_image", "initImage"], ["mask", "mask"], ["control.image", "controlImage"],
      ] as const).map(async ([key]) => {
        const reference = assetReference(image.metadata, key);
        return reference ? bridge.readImage(assetPath(image, reference)) : null;
      }));
      changeSettings({ ...restored, initImage, mask, controlImage });
      setSelected(null); setFullUrl(""); setNotice("Saved settings and input assets restored.");
    } catch (reason) { setNotice(errorText(reason)); }
  };
  const previewControl = async () => {
    if (!settings.controlImage || settings.controlKind === "none") return;
    const token = ++controlToken.current;
    const inputUrl = settings.controlImage.data_url;
    const kind = settings.controlKind;
    setControlBusy(true);
    try {
      const result = await bridge.preprocess(inputUrl, kind);
      if (controlToken.current === token) setControlPreview({ name: `${kind}-map.png`, ...result });
    } catch (reason) { setNotice(errorText(reason)); }
    finally { if (controlToken.current === token) setControlBusy(false); }
  };
  const saveConfig = async (draft: Config) => {
    setLogSecrets(draft.api_key);
    const saved = await bridge.saveConfig(draft);
    controlToken.current += 1; setControlPreview(null); setControlBusy(false);
    setConfig(saved); setServerRequired(false); setServerOpen(false); setNotice("");
    if (!workspace) setWorkspacePrompt(true);
  };
  const checkConfig = async (draft: Config) => {
    setLogSecrets(draft.api_key);
    await bridge.checkServer(draft);
  };
  const confirmDelete = async () => {
    if (!deleteTarget || deleting) return;
    setDeleting(true);
    historyToken.current += 1;
    try {
      if (deleteTarget.image) {
        const image = deleteTarget.image;
        await bridge.deleteImage(deleteTarget.workspacePath, image.session_id, image.id);
        setImages((current) => current.filter((item) => item.id !== image.id || item.session_id !== image.session_id));
        selectionToken.current += 1; setSelected(null); setFullUrl("");
        const token = ++historyToken.current;
        const remaining = await bridge.listSessions();
        if (historyToken.current === token) setSessions(remaining);
      } else if (deleteTarget.session) {
        if (queue.active) throw new Error("Finish or discard queued jobs before deleting a session.");
        const next = await bridge.deleteSession(deleteTarget.workspacePath, deleteTarget.session.session_id);
        setWorkspace(next); selectionToken.current += 1; setSelected(null); setFullUrl("");
        queue.clearFinished();
        await refreshImages();
      }
      setDeleteTarget(null); setNotice("");
    } catch (reason) { setNotice(errorText(reason)); setDeleteTarget(null); }
    finally { setDeleting(false); }
  };
  const groupedComparison = useMemo(() => preview.length > 1 && preview.some((job) => Object.keys(job.context.axes).includes("prompt")) && preview.some((job) => Object.keys(job.context.axes).includes("seed")), [preview]);

  if (loading) return <main className="loading-screen"><div className="brand-mark">x</div><p>Opening Image Studio…</p></main>;
  if (fatal) return <main className="fatal-screen"><p className="eyebrow">Image Studio could not start</p><h1>Startup error</h1><p className="error-banner">{fatal}</p><button className="primary-button" onClick={() => location.reload()}>Try again</button></main>;

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">x</span><div><strong>Image Studio</strong><small>xwen</small></div></div>
      <div className="workspace-controls">
        <label><span className="sr-only">Recent workspace</span><select aria-label="Recent workspace" value={workspace?.path ?? ""} disabled={queue.active} onChange={(event) => void chooseWorkspace(event.target.value)}><option value="" disabled>Choose workspace</option>{config.workspaces.map((path) => <option key={path} value={path}>{path}</option>)}</select></label>
        <button className="quiet-button" disabled={queue.active} onClick={() => void chooseWorkspace()}>Open…</button>
        <button className="quiet-button" disabled={queue.active || !workspace} onClick={() => workspace && void chooseWorkspace(workspace.path, true)}>New session</button>
      </div>
      <button className="icon-button settings-button" disabled={queue.active} onClick={() => setServerOpen(true)} aria-label="Server settings">⚙</button>
    </header>
    {notice && <div className="global-notice" role="status"><span>{notice}</span><button className="icon-button" onClick={() => setNotice("")} aria-label="Dismiss message">×</button></div>}
    <main className="studio-layout">
      <aside className="inspector">
        <SettingsPanel settings={settings} loras={loras} loraLoading={loraLoading} disabled={!workspace} controlPreview={controlPreview} controlBusy={controlBusy} activeDropTarget={imageDrop.activeTarget} onChange={changeSettings} onPick={(kind) => void pickImage(kind).catch((reason: unknown) => setNotice(errorText(reason)))} onClearImage={(kind) => {
          imageDrop.cancel(kind);
          if (kind === "source") changeSettings({ ...settings, initImage: null, mask: null });
          else changeSettings({ ...settings, controlImage: null });
        }} onEditMask={() => setMaskOpen(true)} onRefreshLoras={() => void refreshLoras()} onPreviewControl={() => void previewControl()} onUseControlPreview={() => controlPreview && changeSettings({ ...settings, controlImage: controlPreview, controlKind: "none" })} onGenerate={generate} />
        <BatchPanel settings={settings} axes={axes} mode={batchMode} preview={preview} error={batchError} disabled={!workspace} onAxes={(next) => { setAxes(next); setPreview([]); setBatchError(""); }} onMode={(next) => { setBatchMode(next); setPreview([]); setBatchError(""); }} onPreview={(jobs, error) => { setPreview(jobs); setBatchError(error); }} onQueue={() => {
          if (!workspace || !preview.length) return;
          try { queue.enqueue(preview, workspace.session_id); setNotice(""); }
          catch (reason) { setNotice(errorText(reason)); }
        }} />
        {groupedComparison && <p className="comparison-note">The gallery labels prompt and seed so this matrix stays comparable after rendering.</p>}
      </aside>
      <div className="work-area">
        <Gallery images={images} selected={selected} fullUrl={fullUrl} currentSessionId={workspace?.session_id ?? null} sessions={sessions} deleting={deleting} sessionDeletionDisabled={queue.active} onDelete={(image) => workspace && setDeleteTarget({ workspacePath: workspace.path, image })} onDeleteSession={(session) => workspace && setDeleteTarget({ workspacePath: workspace.path, session })} onSelect={(image) => void chooseSelected(image)} onUseSource={(image) => void useAsSource(image)} onRestore={(image) => void restore(image)} onReveal={(image) => void bridge.reveal(image.path).catch((reason: unknown) => setNotice(errorText(reason)))} />
        <QueuePanel items={queue.items} paused={queue.paused} running={queue.running} pending={queue.pending} onStop={queue.stopAfterCurrent} onResume={queue.resume} onRetry={queue.retry} onClear={queue.clearFinished} onDiscard={queue.discardPending} />
      </div>
    </main>
    {serverOpen && <ServerDialog config={config} configPath={configPath} required={serverRequired} disabled={queue.active} onClose={() => setServerOpen(false)} onSave={saveConfig} onCheck={checkConfig} />}
    {workspacePrompt && !serverOpen && <div className="modal-backdrop" role="presentation"><section className="modal workspace-modal" role="dialog" aria-modal="true" aria-labelledby="workspace-title"><p className="eyebrow">First workspace</p><h2 id="workspace-title">Choose where your sessions live</h2><p className="muted">Each generation is saved with its metadata and local input snapshots. You can switch among recent workspaces later.</p><button className="primary-button full-button" onClick={() => void chooseWorkspace()}>Choose workspace folder</button><p className="field-help">Canceling the folder picker is safe. This window stays here so you can try again.</p></section></div>}
    {maskOpen && settings.initImage && <MaskEditor source={settings.initImage} initial={settings.mask} onCancel={() => setMaskOpen(false)} onSave={(mask) => { changeSettings({ ...settings, mask }); setMaskOpen(false); }} />}
    {deleteTarget && <div className="modal-backdrop delete-backdrop" role="presentation"><section className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-title"><h2 id="delete-title">{deleteTarget.image ? "Delete image?" : "Delete whole session?"}</h2><p className="muted">{deleteTarget.image ? "Permanently delete this image and its YAML generation record. Shared input snapshots remain available to other images." : `Permanently delete session ${deleteTarget.session?.session_id}, including all ${deleteTarget.session?.image_count} images, YAML records and input snapshots. This includes images outside the loaded gallery.`}</p><p>This cannot be undone.</p><div className="modal-actions"><button className="quiet-button" disabled={deleting} onClick={() => setDeleteTarget(null)}>Cancel</button><button className="primary-button danger-button" disabled={deleting} onClick={() => void confirmDelete()}>{deleting ? "Deleting…" : "Delete permanently"}</button></div></section></div>}
  </div>;
}
