import { useState } from "react";
import { PromptGenerator } from "./PromptGenerator";
import type { InputImage, LoraCandidate, StudioSettings } from "../domain";
import type { ImageDropTarget } from "../useImageDrop";

interface Props {
  settings: StudioSettings;
  loras: LoraCandidate[];
  loraLoading: boolean;
  disabled: boolean;
  controlPreview: InputImage | null;
  controlBusy: boolean;
  onChange(settings: StudioSettings): void;
  onPick(kind: "source" | "control"): void;
  onClearImage(kind: ImageDropTarget): void;
  activeDropTarget: ImageDropTarget | null;
  onEditMask(): void;
  onRefreshLoras(): void;
  onPreviewControl(): void;
  onUseControlPreview(): void;
  onGenerate(): void;
}

function NumberField({ label, value, min, max, step = 1, onValue }: { label: string; value: number; min?: number; max?: number; step?: number; onValue(value: number): void }) {
  return <label className="field"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onValue(Number(event.target.value))} /></label>;
}

function ImageInput({ kind, label, image, hint, active, onPick, onClear }: { kind: ImageDropTarget; label: string; image: InputImage | null; hint: string; active: boolean; onPick(): void; onClear(): void }) {
  return <div className={`image-input ${active ? "drop-active" : ""}`} data-image-drop={kind}><span className="field-label">{label}</span>{image ? <div className="image-chip"><img src={image.data_url} alt="" /><span><strong>{image.name}</strong><small>{image.width}×{image.height} · Drop to replace</small></span><button className="icon-button" onClick={onClear} aria-label={`Remove ${label}`}>×</button></div> : <button className="drop-button" onClick={onPick}><span>Choose image</span><small>{hint} · or drop here</small></button>}</div>;
}

export function SettingsPanel({ settings, loras, loraLoading, disabled, controlPreview, controlBusy, activeDropTarget, onChange, onPick, onClearImage, onEditMask, onRefreshLoras, onPreviewControl, onUseControlPreview, onGenerate }: Props) {
  const [loraPath, setLoraPath] = useState("");
  const change = <K extends keyof StudioSettings>(key: K, value: StudioSettings[K]) => onChange({ ...settings, [key]: value });
  const addLora = () => {
    if (!loraPath || settings.loras.some((lora) => lora.name === loraPath)) return;
    change("loras", [...settings.loras, { name: loraPath, weight: 1 }]);
    setLoraPath("");
  };
  return <div className="settings-panel">
    <section className="control-section hero-controls">
      <p className="eyebrow">Create</p>
      <div className="segmented modes" aria-label="Generation mode">
        {([['text', 'Text'], ['img2img', 'Image'], ['inpaint', 'Inpaint']] as const).map(([value, label]) => <button key={value} className={settings.mode === value ? "selected" : ""} onClick={() => onChange({ ...settings, mode: value, strength: value === "inpaint" ? 1 : value === "img2img" ? 0.6 : settings.strength })}>{label}</button>)}
      </div>
      <label className="field prompt-field"><span>Prompt</span><textarea name="prompt" rows={5} placeholder="Describe the image you want to make…" value={settings.prompt} onChange={(event) => change("prompt", event.target.value)} /></label>
      <PromptGenerator prompt={settings.prompt} disabled={disabled} onUse={(prompt) => change("prompt", prompt)} />
    </section>

    {settings.mode !== "text" && <section className="control-section">
      <ImageInput kind="source" label="Source image" image={settings.initImage} hint="PNG or JPEG · dimensions fill automatically" active={activeDropTarget === "source"} onPick={() => onPick("source")} onClear={() => onClearImage("source")} />
      {settings.initImage && <div className="two-columns"><NumberField label="Strength" value={settings.strength} min={0} max={1} step={0.05} onValue={(value) => change("strength", value)} />{settings.mode === "inpaint" && <NumberField label="Mask blur" value={settings.maskBlur} min={0} step={1} onValue={(value) => change("maskBlur", value)} />}</div>}
      {settings.mode === "inpaint" && <div className="mask-control"><span className="field-label">Repaint mask</span><button className={settings.mask ? "mask-ready-button" : "drop-button"} disabled={!settings.initImage} onClick={onEditMask}>{settings.mask ? <><img src={settings.mask.data_url} alt="" /><span><strong>Edit mask</strong><small>White repaints · black keeps</small></span></> : <><span>Paint or import mask</span><small>Choose a source image first</small></>}</button></div>}
    </section>}

    <section className="control-section">
      <div className="section-title"><h3>Output</h3><span>Multiples of 16</span></div>
      <div className="two-columns"><NumberField label="Width" value={settings.width} min={16} max={8192} step={16} onValue={(value) => change("width", value)} /><NumberField label="Height" value={settings.height} min={16} max={8192} step={16} onValue={(value) => change("height", value)} /></div>
      <div className="three-columns"><NumberField label="Steps" value={settings.steps} min={1} max={50} onValue={(value) => change("steps", value)} /><label className="field seed-field"><span>Seed</span><input inputMode="numeric" placeholder="Random" value={settings.seed} onChange={(event) => change("seed", event.target.value)} /></label><NumberField label="Count" value={settings.count} min={1} max={1000} onValue={(value) => change("count", value)} /></div>
      <p className="field-help">Seed may be blank or a safe nonnegative JavaScript integer.</p>
    </section>

    <details className="control-section disclosure">
      <summary><span>LoRAs</span><span className="summary-note">{settings.loras.length || "None"}</span></summary>
      <div className="disclosure-body">
        {settings.loras.map((lora, index) => <div className="lora-row" key={`${lora.name}-${index}`}><span title={lora.name}>{lora.name.split("/").pop()}</span><input aria-label={`LoRA ${index + 1} weight`} type="number" step="0.05" value={lora.weight} onChange={(event) => change("loras", settings.loras.map((item, itemIndex) => itemIndex === index ? { ...item, weight: Number(event.target.value) } : item))} /><button className="icon-button" aria-label={`Remove LoRA ${index + 1}`} onClick={() => change("loras", settings.loras.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}
        <div className="select-action"><select aria-label="Available LoRA" value={loraPath} onChange={(event) => setLoraPath(event.target.value)}><option value="">Choose from server…</option>{loras.map((lora) => <option key={lora.path} value={lora.path}>{lora.name} · {(lora.size_bytes / 1_000_000).toFixed(0)} MB</option>)}</select><button className="quiet-button" disabled={!loraPath} onClick={addLora}>Add</button></div>
        <button className="text-button" onClick={onRefreshLoras}>{loraLoading ? "Refreshing…" : "Refresh server LoRAs"}</button>
      </div>
    </details>

    <details className="control-section disclosure" open={settings.controlEnabled}>
      <summary onClick={(event) => { event.preventDefault(); change("controlEnabled", !settings.controlEnabled); }}><span>ControlNet</span><span className={`switch ${settings.controlEnabled ? "on" : ""}`} aria-hidden="true" /></summary>
      {settings.controlEnabled && <div className="disclosure-body">
        <ImageInput kind="control" label="Control image" image={settings.controlImage} hint="Independent from the source image" active={activeDropTarget === "control"} onPick={() => onPick("control")} onClear={() => onClearImage("control")} />
        <label className="field"><span>Preprocessor</span><select value={settings.controlKind} onChange={(event) => change("controlKind", event.target.value as StudioSettings["controlKind"])}><option value="none">None · image is already a map</option><option value="canny">Canny edges</option><option value="depth">Depth</option><option value="pose">Pose</option></select></label>
        <div className="three-columns"><NumberField label="Scale" value={settings.controlScale} min={0} max={1} step={0.05} onValue={(value) => change("controlScale", value)} /><NumberField label="Start" value={settings.controlStart} min={0} max={1} step={0.05} onValue={(value) => change("controlStart", value)} /><NumberField label="End" value={settings.controlEnd} min={0} max={1} step={0.05} onValue={(value) => change("controlEnd", value)} /></div>
        {settings.controlImage && settings.controlKind !== "none" && <button className="quiet-button full-button" disabled={controlBusy} onClick={onPreviewControl}>{controlBusy ? "Building preview…" : `Preview ${settings.controlKind} map`}</button>}
        {controlPreview && <div className="control-preview"><img src={controlPreview.data_url} alt={`${settings.controlKind} control map preview`} /><button className="quiet-button" onClick={onUseControlPreview}>Use this map</button></div>}
      </div>}
    </details>
    <button className="generate-button" disabled={disabled} onClick={onGenerate}><span>Generate</span><small>{disabled ? "Choose a workspace" : `${settings.count} image${settings.count === 1 ? "" : "s"}`}</small></button>
  </div>;
}
