import type { BatchAxis, BatchMode, LoraCandidate, PlannedJob, StudioSettings } from "../domain";
import { MAX_JOBS, planJobs } from "../domain";

function displayAxisValue(value: string | number): string {
  const text = String(value);
  return text.includes("/") || text.includes("\\") ? text.split(/[\\/]/).filter(Boolean).pop() ?? text : text;
}

interface Props {
  settings: StudioSettings;
  loras: LoraCandidate[];
  axes: BatchAxis[];
  mode: BatchMode;
  preview: PlannedJob[];
  error: string;
  disabled: boolean;
  onAxes(axes: BatchAxis[]): void;
  onMode(mode: BatchMode): void;
  onPreview(jobs: PlannedJob[], error: string): void;
  onQueue(): void;
  onTryAllLoras(): void;
}

export function BatchPanel({ settings, loras, axes, mode, preview, error, disabled, onAxes, onMode, onPreview, onQueue, onTryAllLoras }: Props) {
  const parameters = ["prompt", "seed", "width", "height", "steps", ...(settings.mode === "text" ? [] : ["strength"]), ...(settings.mode === "inpaint" ? ["mask_blur"] : []), ...(settings.controlEnabled ? ["control.scale", "control.start", "control.end"] : []), ...(loras.length ? ["lora"] : []), ...settings.loras.map((_, index) => `loras.${index}.weight`)];
  const refresh = () => {
    try { onPreview(planJobs(settings, axes, mode), ""); }
    catch (reason) { onPreview([], reason instanceof Error ? reason.message : String(reason)); }
  };
  return (
    <details className="batch-card">
      <summary><span>Batch variations</span><span className="summary-note">{preview.length ? `${preview.length} ready` : "Optional"}</span></summary>
      <div className="batch-body">
        <div className="segmented" aria-label="Batch combination mode">
          <button className={mode === "matrix" ? "selected" : ""} onClick={() => onMode("matrix")}>Matrix</button>
          <button className={mode === "paired" ? "selected" : ""} onClick={() => onMode("paired")}>Paired</button>
        </div>
        <p className="field-help">Matrix makes every combination. Paired matches rows; a one-value axis repeats. Maximum {MAX_JOBS} images.</p>
        {axes.map((axis, index) => (
          <div className="axis-row" key={index}>
            <select aria-label={`Batch parameter ${index + 1}`} value={axis.parameter} onChange={(event) => onAxes(axes.map((item, itemIndex) => itemIndex === index ? { ...item, parameter: event.target.value } : item))}>
              {parameters.map((parameter) => <option key={parameter} value={parameter}>{parameter.replace("loras.", "LoRA ").replace(".weight", " weight")}</option>)}
            </select>
            {axis.parameter === "prompt"
              ? <textarea aria-label={`Batch values ${index + 1}`} rows={2} placeholder="One prompt per line" value={axis.values} onChange={(event) => onAxes(axes.map((item, itemIndex) => itemIndex === index ? { ...item, values: event.target.value } : item))} />
              : axis.parameter === "lora"
                ? <textarea aria-label={`Batch values ${index + 1}`} rows={2} placeholder="One LoRA path per line" value={axis.values} onChange={(event) => onAxes(axes.map((item, itemIndex) => itemIndex === index ? { ...item, values: event.target.value } : item))} />
              : <input aria-label={`Batch values ${index + 1}`} placeholder="1, 2, 3 or start:end:step" value={axis.values} onChange={(event) => onAxes(axes.map((item, itemIndex) => itemIndex === index ? { ...item, values: event.target.value } : item))} />}
            <button className="icon-button" aria-label={`Remove batch axis ${index + 1}`} onClick={() => onAxes(axes.filter((_, itemIndex) => itemIndex !== index))}>×</button>
          </div>
        ))}
        <div className="button-row">
          <button className="quiet-button" disabled={!parameters.length} onClick={() => onAxes([...axes, { parameter: parameters.find((parameter) => !axes.some((axis) => axis.parameter === parameter)) ?? parameters[0]!, values: "" }])}>Add axis</button>
          <button className="quiet-button" disabled={!loras.length} onClick={onTryAllLoras}>Try with all LoRAs</button>
          <button className="quiet-button" onClick={refresh}>Preview batch</button>
        </div>
        {error && <p className="error-banner" role="alert">{error}</p>}
        {!!preview.length && <div className="batch-preview" aria-label="Batch preview">
          <strong>{preview.length} image{preview.length === 1 ? "" : "s"}</strong>
          <ol>{preview.slice(0, 6).map((job) => <li key={job.id}><span>{job.request.seed}</span><span>{job.request.width}×{job.request.height}</span><span title={Object.values(job.context.axes).join(" · ")}>{Object.values(job.context.axes).map(displayAxisValue).join(" · ") || "base settings"}</span></li>)}</ol>
          {preview.length > 6 && <p className="field-help">and {preview.length - 6} more</p>}
          <button className="primary-button full-button" disabled={disabled} onClick={onQueue}>Queue exact preview</button>
        </div>}
      </div>
    </details>
  );
}
