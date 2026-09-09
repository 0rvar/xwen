import { invoke, isTauri } from "@tauri-apps/api/core";

export type LogLevel = "debug" | "info" | "warn" | "error";
export interface LogEntry { level: LogLevel; source: string; message: string }

const originalConsole = {
  log: console.log.bind(console), debug: console.debug.bind(console), info: console.info.bind(console),
  warn: console.warn.bind(console), error: console.error.bind(console),
};
const secrets = new Set<string>();
const pending: LogEntry[] = [];
let timer: ReturnType<typeof setTimeout> | undefined;
let sending = false;
let installed = false;
let warned = false;
let dropped = 0;

export function setLogSecrets(...values: string[]): void {
  for (const value of values) if (value) secrets.add(value);
}

function cleanText(text: string): string {
  text = text.replace(/data:image\/[^;,\s]+;base64,[A-Za-z0-9+/=]*/gi, "[image data]")
    .replace(/\bBearer\s+[^\s,;"']+/gi, "Bearer [redacted]");
  for (const secret of [...secrets].sort((a, b) => b.length - a.length)) text = text.replaceAll(secret, "[redacted]");
  return text.slice(0, 8_000);
}

export function formatLogValues(values: unknown[]): string {
  const seen = new WeakSet<object>();
  let remaining = 200;
  const simplify = (value: unknown, depth: number): unknown => {
    if (--remaining < 0) return "[truncated]";
    if (typeof value === "string") return cleanText(value);
    if (typeof value === "bigint" || typeof value === "symbol" || typeof value === "function") return String(value);
    if (value === null || typeof value !== "object") return value;
    if (seen.has(value)) return "[circular]";
    if (depth > 4) return "[nested object]";
    seen.add(value);
    if (value instanceof Error) return { name: value.name, message: cleanText(value.message), stack: cleanText(value.stack ?? ""), cause: simplify(value.cause, depth + 1) };
    if (Array.isArray(value)) return value.slice(0, 30).map((item) => simplify(item, depth + 1));
    const result: Record<string, unknown> = {};
    for (const key of Object.keys(value).slice(0, 30)) {
      if (/api[_-]?key|authorization|password|secret|token|data[_-]?url|b64_json|^prompt$|^init_image$|^mask$|^image$/i.test(key)) result[key] = "[redacted]";
      else {
        try { result[key] = simplify((value as Record<string, unknown>)[key], depth + 1); }
        catch { result[key] = "[unreadable]"; }
      }
    }
    return result;
  };
  try {
    return cleanText(values.slice(0, 30).map((value) => typeof value === "string" ? cleanText(value) : JSON.stringify(simplify(value, 0)) ?? String(value)).join(" "));
  } catch { return "[Could not serialize log message]"; }
}

async function send(entries: LogEntry[]): Promise<void> {
  if (isTauri()) await invoke("frontend_logs", { entries });
  else if (import.meta.env.DEV && import.meta.env.VITE_BRIDGE_MODE === "preview") {
    const preview = globalThis as typeof globalThis & { __XWEN_PREVIEW_LOGS__?: LogEntry[] };
    preview.__XWEN_PREVIEW_LOGS__ = [...(preview.__XWEN_PREVIEW_LOGS__ ?? []), ...entries].slice(-200);
  }
}

export async function flushLogs(): Promise<void> {
  if (timer !== undefined) { clearTimeout(timer); timer = undefined; }
  if (sending || !pending.length) return;
  sending = true;
  const batch = pending.splice(0, dropped ? 49 : 50);
  if (dropped) { batch.unshift({ level: "warn", source: "frontend.logging", message: `${dropped} messages dropped because the log queue was full` }); dropped = 0; }
  try { await send(batch); }
  catch (error) {
    if (!warned) { warned = true; originalConsole.warn("Could not forward frontend logs to Rust", formatLogValues([error])); }
  } finally {
    sending = false;
    if (pending.length && timer === undefined) timer = setTimeout(() => void flushLogs(), 100);
  }
}

export function logMessage(level: LogLevel, source: string, ...args: unknown[]): void {
  if (pending.length >= 200) { pending.shift(); dropped += 1; }
  pending.push({ level, source: source.slice(0, 100), message: formatLogValues(args) });
  if (timer === undefined) timer = setTimeout(() => void flushLogs(), level === "error" ? 0 : 100);
}

export function reportError(source: string, error: unknown): void { logMessage("error", source, error); }

export function installFrontendLogging(): void {
  if (installed) return;
  installed = true;
  for (const method of ["log", "debug", "info", "warn", "error"] as const) {
    console[method] = (...args: unknown[]) => {
      originalConsole[method](...args);
      logMessage(method === "log" ? "info" : method, `frontend.console.${method}`, ...args);
    };
  }
  window.addEventListener("error", (event: Event) => {
    if (event instanceof ErrorEvent) reportError("frontend.uncaught", event.error ?? event.message);
    else reportError("frontend.resource", `Failed to load ${(event.target as Element | null)?.tagName ?? "resource"}`);
  }, true);
  window.addEventListener("unhandledrejection", (event) => reportError("frontend.unhandledrejection", event.reason));
  window.addEventListener("pagehide", () => { logMessage("info", "frontend.lifecycle", "Page closing"); void flushLogs(); });
  logMessage("info", "frontend.lifecycle", "Frontend starting");
}
