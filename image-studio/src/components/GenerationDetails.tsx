interface DetailRow { label: string; value: string }

export interface GenerationDetailData {
  rows: DetailRow[];
  loras: string[] | null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordedNumber(value: unknown): string {
  const number = finite(value);
  return number === null ? "Not recorded" : String(number);
}

function basename(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

export function readGenerationDetails(metadata: Record<string, unknown>): GenerationDetailData {
  const request = record(metadata.request);
  const response = record(metadata.response);
  const hasInit = typeof request?.init_image === "string" && request.init_image.length > 0;
  const hasMask = typeof request?.mask === "string" && request.mask.length > 0;
  const mode = !request ? "Not recorded" : hasMask ? "Inpainting" : hasInit ? "Image to image" : "Text to image";
  const steps = finite(request?.steps) ?? finite(response?.steps);
  const rows: DetailRow[] = [
    { label: "Mode", value: mode },
    { label: "Steps", value: steps === null ? "Not recorded" : String(steps) },
  ];
  if (hasInit) rows.push({ label: "Strength", value: recordedNumber(request?.strength) });
  if (hasMask) rows.push({ label: "Mask blur", value: recordedNumber(request?.mask_blur) });

  const control = record(request?.control);
  if (request) {
    if (!control) rows.push({ label: "ControlNet", value: "None" });
    else {
      const preprocess = typeof control.preprocess === "string" ? control.preprocess : null;
      rows.push({ label: "ControlNet", value: preprocess === "none" ? "Prepared map" : preprocess ? `${preprocess[0]!.toUpperCase()}${preprocess.slice(1)}` : "Not recorded" });
      rows.push({ label: "Control scale", value: recordedNumber(control.scale) });
      rows.push({ label: "Control start", value: recordedNumber(control.start) });
      rows.push({ label: "Control end", value: recordedNumber(control.end) });
    }
  } else rows.push({ label: "ControlNet", value: "Not recorded" });

  const model = typeof response?.model === "string" && response.model.trim() ? response.model : "Not recorded";
  rows.push({ label: "Model", value: model });

  let loras: string[] | null = null;
  if (Array.isArray(request?.loras)) {
    const parsed = request.loras.map((item) => {
      const lora = record(item);
      const name = typeof lora?.name === "string" && lora.name.trim() ? lora.name : null;
      const weight = finite(lora?.weight);
      return name && weight !== null ? `${basename(name)} · ${weight}` : null;
    });
    loras = parsed.every((item) => item !== null) ? parsed as string[] : null;
  }
  return { rows, loras };
}

export function GenerationDetails({ metadata }: { metadata: Record<string, unknown> }) {
  const details = readGenerationDetails(metadata);
  return <section className="generation-details" aria-labelledby="generation-details-title">
    <h3 id="generation-details-title">Generation parameters</h3>
    <dl>{details.rows.map((row) => <div key={row.label}><dt>{row.label}</dt><dd>{row.value}</dd></div>)}
      <div><dt>LoRAs</dt><dd>{details.loras === null ? "Not recorded" : details.loras.length === 0 ? "None" : <ul>{details.loras.map((lora, index) => <li key={`${lora}-${index}`}>{lora}</li>)}</ul>}</dd></div>
    </dl>
  </section>;
}
