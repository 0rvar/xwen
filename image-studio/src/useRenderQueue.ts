import { useCallback, useEffect, useRef, useState } from "react";
import type { PlannedJob, SavedImage } from "./domain";
import { bridge } from "./bridge";

export type QueueStatus = "pending" | "running" | "done" | "error";
export interface QueueItem {
  key: string;
  sessionId: string;
  job: PlannedJob;
  status: QueueStatus;
  outputs: SavedImage[];
  error?: string;
  startedAt?: number;
}

export function useRenderQueue(onOutputs: (images: SavedImage[]) => void) {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [paused, setPaused] = useState(false);
  const runningRef = useRef(false);
  const callbackRef = useRef(onOutputs);
  callbackRef.current = onOutputs;

  useEffect(() => {
    if (paused || runningRef.current) return;
    const next = items.find((item) => item.status === "pending");
    if (!next) return;
    runningRef.current = true;
    setItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "running", error: undefined, startedAt: Date.now() } : item));
    void bridge.render(next.sessionId, next.job.request, next.job.context).then((outputs) => {
      setItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "done", outputs } : item));
      callbackRef.current(outputs);
    }).catch((error: unknown) => {
      setItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "error", error: error instanceof Error ? error.message : String(error) } : item));
    }).finally(() => {
      runningRef.current = false;
      setItems((current) => [...current]);
    });
  }, [items, paused]);

  const enqueue = useCallback((jobs: PlannedJob[], sessionId: string) => {
    const additions = jobs.map((job) => ({
      key: crypto.randomUUID(),
      sessionId,
      job: {
        ...job,
        request: { ...job.request, loras: job.request.loras.map((lora) => ({ ...lora })), ...(job.request.control ? { control: { ...job.request.control } } : {}) },
        context: { ...job.context, axes: { ...job.context.axes } },
      },
      status: "pending" as const,
      outputs: [],
    }));
    setPaused(false);
    setItems((current) => [...current, ...additions]);
  }, []);

  const retry = useCallback((key: string) => {
    setItems((current) => current.map((item) => item.key === key && item.status === "error" ? { ...item, status: "pending", error: undefined } : item));
    setPaused(false);
  }, []);
  const clearFinished = useCallback(() => setItems((current) => current.filter((item) => item.status === "pending" || item.status === "running")), []);
  const discardPending = useCallback(() => setItems((current) => current.filter((item) => item.status !== "pending")), []);
  const running = items.some((item) => item.status === "running");
  const pending = items.filter((item) => item.status === "pending").length;

  return {
    items,
    paused,
    running,
    pending,
    active: running || pending > 0,
    enqueue,
    retry,
    clearFinished,
    discardPending,
    stopAfterCurrent: () => setPaused(true),
    resume: () => setPaused(false),
  };
}
