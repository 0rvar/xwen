export type Mode = 'text' | 'img2img' | 'inpaint';
export type ControlKind = 'none' | 'canny' | 'pose' | 'depth';
export interface Config { server_url: string; api_key: string; workspaces: string[]; last_workspace: string | null }
export interface Workspace { path: string; session_id: string; session_path: string }
export interface Bootstrap { config: Config; config_path: string; workspace: Workspace | null; warning?: string | null }
export interface InputImage { name: string; data_url: string; width: number; height: number }
export interface LoraCandidate { name: string; path: string; size_bytes: number }
export interface Lora { name: string; weight: number }
export interface RenderRequest {
  prompt: string; width: number; height: number; steps: number; seed: number; n: number; loras: Lora[];
  init_image?: string; strength?: number; mask?: string; mask_blur?: number;
  control?: { image: string; preprocess: ControlKind; scale: number; start: number; end: number };
}
export interface SavedImage {
  id: string; path: string; metadata_path: string; data_url: string; seed: number;
  width: number; height: number; session_id: string; prompt: string; metadata: Record<string, unknown>;
}
export interface StudioSettings {
  mode: Mode; prompt: string; width: number; height: number; steps: number; seed: string; count: number;
  initImage: InputImage | null; mask: InputImage | null; strength: number; maskBlur: number;
  loras: Lora[]; controlEnabled: boolean; controlImage: InputImage | null; controlKind: ControlKind;
  controlScale: number; controlStart: number; controlEnd: number;
}
export interface BatchAxis { parameter: string; values: string }
export type BatchMode = 'matrix' | 'paired';
export interface PlannedJob {
  id: string; request: RenderRequest;
  context: { mode: Mode; batch_index: number; axes: Record<string, string | number>; repeat_index: number };
}
export const MAX_JOBS = 1000;
export const DEFAULT_SETTINGS: StudioSettings = {
  mode: 'text', prompt: '', width: 1024, height: 1024, steps: 8, seed: '', count: 1,
  initImage: null, mask: null, strength: 0.6, maskBlur: 0, loras: [],
  controlEnabled: false, controlImage: null, controlKind: 'canny', controlScale: 0.75, controlStart: 0, controlEnd: 0.8,
};

function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function integer(value: number, min: number, max: number, label: string) {
  ensure(Number.isSafeInteger(value) && value >= min && value <= max, `${label} must be a whole number from ${min} to ${max}.`);
}
function between(value: number, min: number, max: number, label: string) {
  ensure(Number.isFinite(value) && value >= min && value <= max, `${label} must be between ${min} and ${max}.`);
}
function randomSeed(): number {
  const words = crypto.getRandomValues(new Uint32Array(2));
  // Keep room for the largest repeat count below Number.MAX_SAFE_INTEGER.
  return ((words[0]! & 0xfffff) * 4294967296 + words[1]!) % (Number.MAX_SAFE_INTEGER - MAX_JOBS);
}
function seedFromText(value: string): number {
  if (!value.trim()) return randomSeed();
  ensure(/^\d+$/.test(value.trim()), 'Seed must be a nonnegative whole number, or blank for random.');
  const seed = Number(value);
  integer(seed, 0, Number.MAX_SAFE_INTEGER, 'Seed');
  return seed;
}

/** Match the server's squared-displacement, area, width, height ordering. */
export function snapSize(width: number, height: number): { width: number; height: number } {
  integer(width, 1, 8192, 'Source width');
  integer(height, 1, 8192, 'Source height');
  let best: [number, number, number, number] | undefined;
  for (let rows = 1; rows <= 512; rows++) {
    for (let cols = 1; cols <= 512; cols++) {
      if ((rows * cols) % 32 !== 0) continue;
      const w = cols * 16, h = rows * 16;
      const rank: [number, number, number, number] = [(w - width) ** 2 + (h - height) ** 2, w * h, w, h];
      if (!best || rank[0] < best[0] || (rank[0] === best[0] && (rank[1] < best[1] || (rank[1] === best[1] && (w < best[2] || (w === best[2] && h < best[3])))))) best = rank;
    }
  }
  return { width: best![2], height: best![3] };
}

export function validateRequest(request: RenderRequest): void {
  ensure(request.prompt.trim(), 'Enter a prompt.');
  integer(request.width, 16, 8192, 'Width');
  integer(request.height, 16, 8192, 'Height');
  ensure(request.width % 16 === 0 && request.height % 16 === 0, 'Width and height must be multiples of 16.');
  ensure(((request.width / 16) * (request.height / 16)) % 32 === 0, 'The image token count (width / 16 × height / 16) must be a multiple of 32.');
  integer(request.steps, 1, 50, 'Steps');
  integer(request.seed, 0, Number.MAX_SAFE_INTEGER, 'Seed');
  integer(request.n, 1, 4, 'Images per request');
  integer(request.seed + request.n - 1, 0, Number.MAX_SAFE_INTEGER, 'Last image seed');
  if (request.strength !== undefined) {
    ensure(request.init_image, 'Strength requires a source image.');
    between(request.strength, 0, 1, 'Strength');
  }
  if (request.mask) ensure(request.init_image, 'A mask requires a source image.');
  if (request.mask_blur !== undefined) {
    ensure(request.mask, 'Mask blur requires a mask.');
    ensure(Number.isFinite(request.mask_blur) && request.mask_blur >= 0, 'Mask blur must be finite and nonnegative.');
  }
  for (const lora of request.loras) {
    ensure(lora.name.trim(), 'Each LoRA needs a server path or name.');
    ensure(Number.isFinite(lora.weight), 'LoRA weights must be finite.');
  }
  if (request.control) {
    ensure(request.control.image, 'Choose a control image.');
    ensure(['none', 'canny', 'pose', 'depth'].includes(request.control.preprocess), 'Unknown control preprocessor.');
    between(request.control.scale, 0, 1, 'Control scale');
    between(request.control.start, 0, 1, 'Control start');
    between(request.control.end, 0, 1, 'Control end');
    ensure(request.control.start <= request.control.end, 'Control start must be at or before control end.');
  }
}

function rawRequest(settings: StudioSettings, seed: number): RenderRequest {
  const request: RenderRequest = {
    prompt: settings.prompt, width: settings.width, height: settings.height, steps: settings.steps,
    seed, n: 1, loras: settings.loras.map(lora => ({ ...lora })),
  };
  if (settings.mode !== 'text') {
    ensure(settings.initImage, 'Choose a source image.');
    request.init_image = settings.initImage.data_url;
    request.strength = settings.strength;
  }
  if (settings.mode === 'inpaint') {
    ensure(settings.mask, 'Paint or import a mask. White areas will be repainted.');
    request.mask = settings.mask.data_url;
    request.mask_blur = settings.maskBlur;
  }
  if (settings.controlEnabled) {
    ensure(settings.controlImage, 'Choose a control image.');
    request.control = {
      image: settings.controlImage.data_url, preprocess: settings.controlKind,
      scale: settings.controlScale, start: settings.controlStart, end: settings.controlEnd,
    };
  }
  return request;
}

export function buildRequest(settings: StudioSettings): RenderRequest {
  const request = rawRequest(settings, seedFromText(settings.seed));
  validateRequest(request);
  return request;
}

export function parseNumericValues(text: string): number[] {
  ensure(text.trim(), 'Enter values for each batch parameter.');
  const parse = (part: string) => {
    ensure(part.trim() && /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(part.trim()), `Invalid numeric value: ${part || '(empty)'}.`);
    const value = Number(part);
    ensure(Number.isFinite(value), 'Batch values must be finite.');
    return value;
  };
  if (!text.includes(':')) {
    const parts = text.split(',');
    ensure(parts.length <= MAX_JOBS, `An axis may contain at most ${MAX_JOBS} values.`);
    return parts.map(parse);
  }
  const parts = text.split(':');
  ensure(parts.length === 3, 'Use start:end:step for a numeric range, for example 0.4:0.8:0.1.');
  const [start, end, step] = parts.map(parse) as [number, number, number];
  ensure(step !== 0, 'Range step cannot be zero.');
  ensure(start === end || Math.sign(end - start) === Math.sign(step), 'Range step must move from start toward end.');
  const distance = (end - start) / step;
  ensure(Number.isFinite(distance) && distance < MAX_JOBS, `An axis may contain at most ${MAX_JOBS} values.`);
  const count = Math.floor(distance + 1e-9) + 1;
  ensure(count <= MAX_JOBS, `An axis may contain at most ${MAX_JOBS} values.`);
  // Round the arithmetic error, not the user's integers (seeds can need all 16 digits).
  const values = Array.from({ length: count }, (_, i) => {
    if (i === 0) return start;
    const value = start + i * step;
    return Number.isSafeInteger(start) && Number.isSafeInteger(step) ? value : Number(value.toPrecision(15));
  });
  ensure(values.every((value, i) => i === 0 || Math.sign(value - values[i - 1]!) === Math.sign(step)), 'This range step is too small to represent at these values.');
  return values;
}

function axisValues(axis: BatchAxis): (string | number)[] {
  if (axis.parameter === 'lora') {
    const values = axis.values.split('\n').map(line => line.trim()).filter(Boolean);
    ensure(values.length > 0 && values.length <= MAX_JOBS, `Enter 1–${MAX_JOBS} LoRA paths, one per line.`);
    return values;
  }
  if (axis.parameter !== 'prompt') return parseNumericValues(axis.values);
  let values: unknown;
  if (axis.values.trim().startsWith('[')) {
    try { values = JSON.parse(axis.values); } catch { throw new Error('Prompt values must be a JSON string array or one prompt per line.'); }
  } else values = axis.values.split('\n').map(line => line.trim()).filter(Boolean);
  ensure(Array.isArray(values) && values.length > 0 && values.length <= MAX_JOBS && values.every(value => typeof value === 'string' && value.trim()), 'Enter 1–1000 nonempty prompts, one per line or as a JSON string array.');
  return values as string[];
}

function applyAxis(request: RenderRequest, parameter: string, value: string | number): void {
  if (parameter === 'prompt') { request.prompt = String(value); return; }
  if (parameter === 'lora') {
    ensure(typeof value === 'string' && value.trim(), 'LoRA values must be nonempty paths.');
    request.loras = [{ name: value, weight: request.loras[0]?.weight ?? 1 }];
    return;
  }
  ensure(typeof value === 'number', `${parameter} requires numeric values.`);
  if (['seed', 'width', 'height', 'steps'].includes(parameter)) {
    (request as unknown as Record<string, unknown>)[parameter] = value;
  } else if (parameter === 'strength') {
    ensure(request.init_image, 'A strength axis requires img2img or inpainting.');
    request.strength = value;
  } else if (parameter === 'mask_blur') {
    ensure(request.mask, 'A mask blur axis requires inpainting.');
    request.mask_blur = value;
  } else if (/^control\.(scale|start|end)$/.test(parameter)) {
    ensure(request.control, 'Control axes require ControlNet to be enabled.');
    request.control[parameter.split('.')[1] as 'scale' | 'start' | 'end'] = value;
  } else {
    const match = /^loras\.(\d+)\.weight$/.exec(parameter);
    ensure(match, `Unknown batch parameter: ${parameter}.`);
    const lora = request.loras[Number(match[1])];
    ensure(lora, `Add LoRA ${Number(match[1]) + 1} before varying its weight.`);
    lora.weight = value;
  }
}

export function planJobs(settings: StudioSettings, axes: BatchAxis[] = [], mode: BatchMode = 'matrix'): PlannedJob[] {
  integer(settings.count, 1, MAX_JOBS, 'Images per combination');
  ensure(mode === 'matrix' || mode === 'paired', 'Unknown batch mode.');
  ensure(new Set(axes.map(axis => axis.parameter)).size === axes.length, 'Each batch parameter may appear only once.');
  // A seed axis supplies the seed even when the ordinary seed field is empty.
  const base = rawRequest(settings, axes.some(axis => axis.parameter === 'seed') ? 0 : seedFromText(settings.seed));
  const parsed = axes.map(axis => ({ parameter: axis.parameter, values: axisValues(axis) }));
  let combinations: Record<string, string | number>[];
  if (mode === 'paired') {
    const length = Math.max(1, ...parsed.map(axis => axis.values.length));
    ensure(length * settings.count <= MAX_JOBS, `A batch may contain at most ${MAX_JOBS} images.`);
    ensure(parsed.every(axis => axis.values.length === 1 || axis.values.length === length), 'Paired axes must have the same number of values, or one value to reuse for every row.');
    combinations = Array.from({ length }, (_, i) => Object.fromEntries(parsed.map(axis => [axis.parameter, axis.values[axis.values.length === 1 ? 0 : i]!] )));
  } else {
    let total = settings.count;
    for (const axis of parsed) { total *= axis.values.length; ensure(total <= MAX_JOBS, `A batch may contain at most ${MAX_JOBS} images.`); }
    combinations = [{}];
    for (const axis of parsed) combinations = combinations.flatMap(previous => axis.values.map(value => ({ ...previous, [axis.parameter]: value })));
  }
  const jobs: PlannedJob[] = [];
  for (const [batchIndex, values] of combinations.entries()) {
    for (let repeat = 0; repeat < settings.count; repeat++) {
      // Data URLs are immutable strings; only the small editable objects need copies.
      const request = { ...base, loras: base.loras.map(lora => ({ ...lora })), ...(base.control ? { control: { ...base.control } } : {}) };
      for (const [parameter, value] of Object.entries(values)) applyAxis(request, parameter, value);
      request.seed += repeat;
      try { validateRequest(request); } catch (error) { throw new Error(`Combination ${batchIndex + 1}, image ${repeat + 1}: ${error instanceof Error ? error.message : error}`); }
      jobs.push({ id: crypto.randomUUID(), request, context: { mode: settings.mode, batch_index: batchIndex, axes: { ...values }, repeat_index: repeat } });
    }
  }
  return jobs;
}

/** Restore scalar controls; callers load the session's referenced input assets separately. */
export function restoreSettings(metadata: Record<string, unknown>, current: StudioSettings = DEFAULT_SETTINGS): StudioSettings {
  const request = metadata.request as Partial<RenderRequest> | undefined;
  ensure(request && typeof request === 'object', 'This image has no saved generation request.');
  return {
    ...current, mode: request.mask ? 'inpaint' : request.init_image ? 'img2img' : 'text',
    prompt: request.prompt ?? '', width: request.width ?? 1024, height: request.height ?? 1024,
    steps: request.steps ?? 8, seed: String(request.seed ?? ''), count: 1,
    strength: request.strength ?? (request.mask ? 1 : 0.6), maskBlur: request.mask_blur ?? 0,
    loras: (request.loras ?? []).map(lora => ({ ...lora })), controlEnabled: !!request.control,
    controlKind: request.control?.preprocess ?? 'none', controlScale: request.control?.scale ?? 0.75,
    controlStart: request.control?.start ?? 0, controlEnd: request.control?.end ?? 0.8,
    initImage: null, mask: null, controlImage: null,
  };
}
