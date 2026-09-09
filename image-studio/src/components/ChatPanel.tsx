import { useMemo, useState } from "react";
import { bridge } from "../bridge";
import { validateRequest, type RenderRequest, type StudioSettings } from "../domain";
import { reportError } from "../logging";
import type { LoraCandidate } from "../domain";

type ChatMessage = { role: string; content?: unknown; [key: string]: unknown };

const queueTool = {
  type: "function",
  function: {
    name: "queue_txt2img",
    description: "Queue one or more text-to-image generations. Use only for images, never edits or control images.",
    parameters: {
      type: "object",
      additionalProperties: false,
      required: ["prompt"],
      properties: {
        prompt: { type: "string", description: "The complete image prompt." },
        width: { type: "integer", minimum: 16, maximum: 8192 },
        height: { type: "integer", minimum: 16, maximum: 8192 },
        steps: { type: "integer", minimum: 1, maximum: 50 },
        seed: { type: "integer", minimum: 0 },
        n: { type: "integer", minimum: 1, maximum: 4 },
        loras: { type: "array", items: { type: "object", additionalProperties: false, required: ["path", "weight"], properties: { path: { type: "string" }, weight: { type: "number" } } } },
      },
    },
  },
};

function randomSeed(): number {
  const words = crypto.getRandomValues(new Uint32Array(2));
  return ((words[0]! & 0xfffff) * 4294967296 + words[1]!) % (Number.MAX_SAFE_INTEGER - 1000);
}

function textContent(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export function ChatPanel({ settings, loras, sessionId, disabled, onQueue }: { settings: StudioSettings; loras: LoraCandidate[]; sessionId: string | null; disabled: boolean; onQueue(requests: RenderRequest[]): Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const available = useMemo(() => loras.map((lora) => `${lora.name}: ${lora.path}`).join("\n") || "No LoRAs are currently available.", [loras]);

  const send = async () => {
    if (!input.trim() || busy || !sessionId) return;
    const user: ChatMessage = { role: "user", content: input.trim() };
    const system: ChatMessage = { role: "system", content: `You are an image-generation assistant for xwen Image Studio. Discuss ideas naturally, then use queue_txt2img when the user asks to make images. Only use txt2img. Available defaults: width ${settings.width}, height ${settings.height}, steps ${settings.steps}. Valid dimensions are multiples of 16 and the image token count must be a multiple of 32. Available LoRAs (use their exact path and a weight between -4 and 4):\n${available}` };
    let history = [...messages.filter((message) => message.role !== "system"), user];
    setMessages(history); setInput(""); setBusy(true); setError("");
    try {
      for (let round = 0; round < 4; round += 1) {
        const response = await bridge.chat([system, ...history], [queueTool]);
        const choice = (response.choices as Array<Record<string, unknown>> | undefined)?.[0];
        const assistant = choice?.message as ChatMessage | undefined;
        if (!assistant) throw new Error("The server returned no assistant message.");
        history = [...history, assistant];
        const calls = Array.isArray(assistant.tool_calls) ? assistant.tool_calls as ChatMessage[] : [];
        if (!calls.length) break;
        for (const call of calls) {
          const fn = call.function as { name?: unknown; arguments?: unknown } | undefined;
          if (fn?.name !== "queue_txt2img") throw new Error(`Unsupported tool: ${String(fn?.name ?? "unknown")}`);
          const parsed = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : fn.arguments;
          const value = parsed as Record<string, unknown>;
          const request: RenderRequest = {
            prompt: String(value.prompt ?? ""), width: Number(value.width ?? settings.width), height: Number(value.height ?? settings.height),
            steps: Number(value.steps ?? settings.steps), seed: Number(value.seed ?? randomSeed()), n: Number(value.n ?? 1),
            loras: Array.isArray(value.loras) ? value.loras.map((lora) => { const item = lora as Record<string, unknown>; return { name: String(item.path ?? ""), weight: Number(item.weight ?? 0.8) }; }) : [],
          };
          validateRequest(request);
          await onQueue([request]);
          history = [...history, { role: "tool", tool_call_id: call.id, content: JSON.stringify({ queued: true, count: request.n, prompt: request.prompt }) }];
        }
      }
      setMessages(history);
    } catch (reason) { reportError("frontend.chat", reason); setError(reason instanceof Error ? reason.message : String(reason)); setMessages(history); }
    finally { setBusy(false); }
  };

  return <>
    <button className="quiet-button" disabled={disabled} onClick={() => setOpen(true)}>Chat with image assistant</button>
    {open && <div className="modal-backdrop" role="presentation"><section className="modal chat-modal" role="dialog" aria-modal="true" aria-labelledby="chat-title">
      <div className="modal-heading"><div><p className="eyebrow">xwen</p><h2 id="chat-title">Image assistant</h2></div><button className="icon-button" onClick={() => setOpen(false)} aria-label="Close chat">×</button></div>
      <div className="chat-history" aria-live="polite">{messages.length === 0 && <p className="muted">Ask for an image, a few variations, or a LoRA comparison. The assistant can queue text-to-image jobs for you.</p>}{messages.filter((message) => message.role !== "tool" && message.role !== "system").map((message, index) => <div className={`chat-message ${message.role}`} key={index}><strong>{message.role === "user" ? "You" : "Assistant"}</strong><p>{textContent(message.content) || (message.tool_calls ? "Queued a generation." : "")}</p></div>)}</div>
      <label className="field"><span>Message</span><textarea rows={3} value={input} disabled={busy || disabled} placeholder="Make a warm terracotta garden pavilion…" onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) void send(); }} /></label>
      {error && <p className="error-banner" role="alert">{error}</p>}
      <div className="dialog-actions"><span className="field-help">⌘↵ to send</span><button className="quiet-button" onClick={() => setMessages([])}>Clear</button><button className="primary-button" disabled={busy || !input.trim() || !sessionId} onClick={() => void send()}>{busy ? "Thinking…" : "Send"}</button></div>
    </section></div>}
  </>;
}
