import { buildRequest, MAX_JOBS, validateRequest, type LoraCandidate, type RenderRequest, type StudioSettings } from "./domain";

export type ChatMessage = { role: "user" | "assistant" | "tool" | "system"; content?: string | null; [key: string]: unknown };
export const MAX_CHAT_ROUNDS = 16;
export type ChatDefaults = RenderRequest & { advanceSeed?: boolean };

export const queueTool = {
  type: "function",
  function: {
    name: "queue_txt2img",
    description: "Stage a batch of text-to-image jobs. All jobs are submitted together after your final reply; images render one at a time. Include every requested prompt, variation and comparison; never use for edits or control images.",
    parameters: {
      type: "object", additionalProperties: false, required: ["jobs"],
      properties: {
        jobs: {
          type: "array", minItems: 1, maxItems: MAX_JOBS,
          items: {
            type: "object", additionalProperties: false, required: ["prompt"],
            properties: {
              prompt: { type: "string", minLength: 1, description: "The complete image prompt." },
              width: { type: "integer", minimum: 16, maximum: 8192 },
              height: { type: "integer", minimum: 16, maximum: 8192 },
              steps: { type: "integer", minimum: 1, maximum: 50 },
              seed: { type: "integer", minimum: 0, maximum: Number.MAX_SAFE_INTEGER },
              n: { type: "integer", minimum: 1, maximum: MAX_JOBS, description: "Number of images with consecutive seeds for this prompt." },
              loras: { type: "array", items: { type: "object", additionalProperties: false, required: ["path", "weight"], properties: { path: { type: "string" }, weight: { type: "number", minimum: -4, maximum: 4 } } } },
            },
          },
        },
      },
    },
  },
};

function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function object(value: unknown, label: string): Record<string, unknown> {
  ensure(value !== null && typeof value === "object" && !Array.isArray(value), `${label} must be an object.`);
  return value as Record<string, unknown>;
}
function fields(value: Record<string, unknown>, allowed: string[], label: string) {
  ensure(Object.keys(value).every((key) => allowed.includes(key)), `${label} contains unsupported fields.`);
}

export function captureChatDefaults(settings: StudioSettings): ChatDefaults {
  return { ...buildRequest({ ...settings, mode: "text", prompt: "Chat defaults", controlEnabled: false }), advanceSeed: !settings.seed.trim() };
}

export function chatSystemMessage(defaults: RenderRequest, loras: LoraCandidate[]): ChatMessage {
  const available = loras.map((lora) => `${lora.name}: ${lora.path}`).join("\n") || "No LoRAs are currently available.";
  return { role: "system", content: `You are an image-generation assistant for xwen Image Studio. Discuss ideas naturally. When the user explicitly asks to generate, make, render or queue images, act immediately with queue_txt2img; do not ask for redundant confirmation. Use the captured defaults for unspecified parameters. Fulfill the complete requested count, all distinct prompts, variations and LoRA comparisons using the tool's jobs array, with more calls if needed. Each job n is its image count; repetitions use consecutive seeds. All jobs remain staged in memory until you finish the whole turn with a concise final summary. Tool results mean staged, not queued or rendered. After all tools, finish your reply so the client can submit the complete batch. Do not claim jobs are already queued or rendered. Only use txt2img; edits and control images are unsupported. At most ${MAX_JOBS} images per turn. Defaults: width ${defaults.width}, height ${defaults.height}, steps ${defaults.steps}, seed ${defaults.seed}, n 1, LoRAs ${JSON.stringify(defaults.loras.map((lora) => ({ path: lora.name, weight: lora.weight })))}. Valid dimensions are multiples of 16 and (width / 16 × height / 16) must be a multiple of 32. Available LoRAs (use only their exact path and a finite weight between -4 and 4, or [] for no LoRAs):\n${available}` };
}

export function normalizeChatJobs(argumentsValue: unknown, defaults: ChatDefaults, loras: LoraCandidate[], remaining = MAX_JOBS, seedOffset = 0): RenderRequest[] {
  const parsed: unknown = typeof argumentsValue === "string" ? JSON.parse(argumentsValue) : argumentsValue;
  const batch = object(parsed, "Tool arguments");
  fields(batch, ["jobs"], "Tool arguments");
  ensure(Array.isArray(batch.jobs) && batch.jobs.length > 0, "queue_txt2img requires a nonempty jobs array.");
  ensure(batch.jobs.length <= remaining, `A turn may contain at most ${MAX_JOBS} images.`);
  const available = new Set(loras.map((lora) => lora.path));
  const requests: RenderRequest[] = [];
  for (const item of batch.jobs) {
    const value = object(item, "Job");
    fields(value, ["prompt", "width", "height", "steps", "seed", "n", "loras"], "Job");
    ensure(typeof value.prompt === "string", "Each job requires a prompt string.");
    const numeric = (key: "width" | "height" | "steps" | "seed" | "n", fallback: number) => {
      if (!(key in value)) return fallback;
      ensure(typeof value[key] === "number", `${key} must be a number.`);
      return value[key];
    };
    const loraValues = "loras" in value ? value.loras : defaults.loras.map((lora) => ({ path: lora.name, weight: lora.weight }));
    ensure(Array.isArray(loraValues), "LoRAs must be an array.");
    const request: RenderRequest = {
      prompt: value.prompt, width: numeric("width", defaults.width), height: numeric("height", defaults.height),
      steps: numeric("steps", defaults.steps), seed: numeric("seed", defaults.seed + (defaults.advanceSeed ? seedOffset + requests.length : 0)), n: 1,
      loras: loraValues.map((entry) => {
        const lora = object(entry, "LoRA");
        fields(lora, ["path", "weight"], "LoRA");
        ensure(typeof lora.path === "string" && available.has(lora.path), "Each LoRA must use an exact path from the current server list.");
        ensure(typeof lora.weight === "number" && Number.isFinite(lora.weight) && lora.weight >= -4 && lora.weight <= 4, "LoRA weights must be finite numbers between -4 and 4.");
        return { name: lora.path, weight: lora.weight };
      }),
    };
    const count = numeric("n", 1);
    ensure(Number.isSafeInteger(count) && count >= 1 && count <= MAX_JOBS, `n must be a whole number from 1 to ${MAX_JOBS}.`);
    validateRequest(request);
    ensure(request.seed <= Number.MAX_SAFE_INTEGER - (count - 1), "The last image seed exceeds the safe integer range.");
    ensure(requests.length + count <= remaining, `A turn may contain at most ${MAX_JOBS} images.`);
    for (let index = 0; index < count; index++) requests.push({ ...request, seed: request.seed + index, n: 1, loras: request.loras.map((lora) => ({ ...lora })) });
  }
  return requests;
}

export async function runChatTurn({ history, defaults, loras, chat, onQueue, onStaged }: {
  history: ChatMessage[];
  defaults: ChatDefaults;
  loras: LoraCandidate[];
  chat(messages: unknown[], tools: unknown[]): Promise<Record<string, unknown>>;
  onQueue(requests: RenderRequest[]): Promise<void>;
  onStaged?(count: number): void;
}): Promise<{ history: ChatMessage[]; queued: number }> {
  const capturedDefaults = structuredClone(defaults);
  const capturedLoras = structuredClone(loras);
  const transcript = structuredClone(history.filter((message) => message.role !== "system"));
  const system = chatSystemMessage(capturedDefaults, capturedLoras);
  const staged: RenderRequest[] = [];
  const callIds = new Set<string>();
  for (let round = 0; round < MAX_CHAT_ROUNDS; round++) {
    const response = await chat([system, ...transcript], [queueTool]);
    ensure(Array.isArray(response.choices) && response.choices.length === 1, "The server must return exactly one assistant choice.");
    const choice = object(response.choices[0], "Assistant choice");
    ensure(choice.finish_reason !== "length", "The assistant reached its output token limit. Try a smaller batch.");
    ensure(choice.finish_reason === "stop" || choice.finish_reason === "tool_calls", `The assistant did not finish successfully (${String(choice.finish_reason ?? "missing finish reason")}).`);
    const raw = object(choice.message, "Assistant message");
    ensure(raw.role === "assistant", "The server returned an invalid assistant role.");
    ensure(raw.content === undefined || raw.content === null || typeof raw.content === "string", "The assistant returned malformed text.");
    ensure(raw.tool_calls === undefined || Array.isArray(raw.tool_calls), "The assistant returned malformed tool calls.");
    const calls = (raw.tool_calls ?? []) as unknown[];
    const assistant: ChatMessage = { role: "assistant", content: raw.content as string | null | undefined, ...(calls.length ? { tool_calls: calls } : {}) };
    transcript.push(assistant);
    if (!calls.length) {
      ensure(choice.finish_reason === "stop", "The assistant requested tools but supplied no calls.");
      ensure(typeof raw.content === "string" && raw.content.trim(), "The assistant returned no final reply.");
      if (staged.length) await onQueue(staged);
      return { history: transcript, queued: staged.length };
    }
    for (const rawCall of calls) {
      const call = object(rawCall, "Tool call");
      ensure(call.type === "function" && typeof call.id === "string" && call.id.trim(), "The assistant returned an invalid tool call.");
      ensure(!callIds.has(call.id), "The assistant reused a tool call ID.");
      callIds.add(call.id);
      const fn = object(call.function, "Tool function");
      ensure(fn.name === "queue_txt2img", `Unsupported tool: ${String(fn.name ?? "unknown")}.`);
      const jobs = normalizeChatJobs(fn.arguments, capturedDefaults, capturedLoras, MAX_JOBS - staged.length, staged.length);
      staged.push(...jobs);
      transcript.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify({ staged: true, count: jobs.length, total_staged: staged.length, status: "Waiting for your final reply before queue submission." }) });
      onStaged?.(staged.length);
    }
  }
  throw new Error(`The assistant exceeded ${MAX_CHAT_ROUNDS} rounds without a final reply. Try a smaller batch.`);
}
