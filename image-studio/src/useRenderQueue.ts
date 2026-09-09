import { useCallback, useEffect, useRef, useState } from "react";
import { MAX_JOBS, type PlannedJob, type SavedImage } from "./domain";
import { bridge } from "./bridge";
import { createBatchDraft, type BatchDefinition } from "./batchDraft";
import { reportError } from "./logging";

export type QueueStatus = "pending" | "running" | "done" | "error";
export interface QueueItem {
  key: string;
  sessionId: string;
  batchId: string;
  jobId: string;
  job: PlannedJob;
  status: QueueStatus;
  outputs: SavedImage[];
  error?: string;
  startedAt?: number;
}

function copyJob(job: PlannedJob): PlannedJob {
  return {
    ...job,
    request: {
      ...job.request,
      loras: job.request.loras.map((lora) => ({ ...lora })),
      ...(job.request.control ? { control: { ...job.request.control } } : {}),
    },
    context: { ...job.context, axes: { ...job.context.axes } },
  };
}

export function useRenderQueue(onOutputs: (images: SavedImage[]) => void) {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [paused, setPausedState] = useState(false);
  const [preparing, setPreparing] = useState(0);
  const [discarding, setDiscarding] = useState(false);
  const itemsRef = useRef<QueueItem[]>([]);
  const pausedRef = useRef(false);
  const preparingRef = useRef(0);
  const runningRef = useRef(false);
  const discardingRef = useRef(false);
  const preparationTail = useRef<Promise<void>>(Promise.resolve());
  const callbackRef = useRef(onOutputs);
  callbackRef.current = onOutputs;

  const replaceItems = useCallback((update: (current: QueueItem[]) => QueueItem[]) => {
    const next = update(itemsRef.current);
    itemsRef.current = next;
    setItems(next);
  }, []);
  const setPaused = useCallback((next: boolean) => {
    pausedRef.current = next;
    setPausedState(next);
  }, []);

  useEffect(() => {
    if (pausedRef.current || discardingRef.current || paused || discarding || runningRef.current) return;
    const next = itemsRef.current.find((item) => item.status === "pending");
    if (!next) return;
    runningRef.current = true;
    replaceItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "running", error: undefined, startedAt: Date.now() } : item));
    void bridge.renderBatchJob(next.sessionId, next.batchId, next.jobId).then((outputs) => {
      replaceItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "done", outputs } : item));
      callbackRef.current(outputs);
    }).catch((error: unknown) => {
      reportError("frontend.render", error);
      replaceItems((current) => current.map((item) => item.key === next.key ? { ...item, status: "error", error: error instanceof Error ? error.message : String(error) } : item));
    }).finally(() => {
      runningRef.current = false;
      replaceItems((current) => [...current]);
    });
  }, [discarding, items, paused, replaceItems]);

  const enqueue = useCallback((jobs: PlannedJob[], sessionId: string, definition: BatchDefinition): Promise<void> => {
    if (!jobs.length) throw new Error("A batch must contain at least one image.");
    const unfinished = itemsRef.current.filter((item) => item.status !== "done").length + preparingRef.current;
    if (unfinished + jobs.length > MAX_JOBS) throw new Error(`The queue may contain at most ${MAX_JOBS} unfinished images.`);

    const displayJobs = jobs.map(copyJob);
    const draft = createBatchDraft(jobs, definition);
    const startsNewRun = preparingRef.current === 0 && !itemsRef.current.some((item) => item.status === "pending" || item.status === "running");
    preparingRef.current += jobs.length;
    setPreparing(preparingRef.current);
    if (startsNewRun) setPaused(false);

    const prepare = preparationTail.current.then(async () => {
      const handle = await bridge.createBatch(sessionId, draft);
      if (!handle.id || handle.job_ids.length !== displayJobs.length || handle.job_ids.some((id) => !id)) throw new Error("The batch manifest returned an invalid job list.");
      replaceItems((current) => [...current, ...displayJobs.map((job, index) => ({
        key: crypto.randomUUID(),
        sessionId,
        batchId: handle.id,
        jobId: handle.job_ids[index]!,
        job,
        status: "pending" as const,
        outputs: [],
      }))]);
    }).finally(() => {
      preparingRef.current -= jobs.length;
      setPreparing(preparingRef.current);
    });
    preparationTail.current = prepare.catch(() => undefined);
    return prepare;
  }, [replaceItems, setPaused]);

  const retry = useCallback((key: string) => {
    if (discardingRef.current) return;
    replaceItems((current) => current.map((item) => item.key === key && item.status === "error" ? { ...item, status: "pending", error: undefined } : item));
    setPaused(false);
  }, [replaceItems, setPaused]);
  const clearFinished = useCallback(() => replaceItems((current) => current.filter((item) => item.status === "pending" || item.status === "running")), [replaceItems]);
  const discardPending = useCallback(async () => {
    if (discardingRef.current || preparingRef.current > 0) return;
    setPaused(true);
    discardingRef.current = true;
    setDiscarding(true);
    const pendingItems = itemsRef.current.filter((item) => item.status === "pending");
    const groups = new Map<string, { sessionId: string; batchId: string; keys: string[]; jobIds: string[] }>();
    for (const item of pendingItems) {
      const groupKey = `${item.sessionId}\u0000${item.batchId}`;
      const group = groups.get(groupKey) ?? { sessionId: item.sessionId, batchId: item.batchId, keys: [], jobIds: [] };
      group.keys.push(item.key);
      group.jobIds.push(item.jobId);
      groups.set(groupKey, group);
    }
    const failures: unknown[] = [];
    for (const group of groups.values()) {
      try {
        await bridge.discardBatchJobs(group.sessionId, group.batchId, group.jobIds);
        const removed = new Set(group.keys);
        replaceItems((current) => current.filter((item) => !removed.has(item.key)));
      } catch (error) {
        reportError("frontend.discard-batch", error);
        failures.push(error);
      }
    }
    discardingRef.current = false;
    setDiscarding(false);
    if (failures.length) throw failures[0];
  }, [replaceItems, setPaused]);
  const running = items.some((item) => item.status === "running");
  const pending = items.filter((item) => item.status === "pending").length;

  return {
    items,
    paused,
    running,
    pending,
    preparing,
    discarding,
    active: preparing > 0 || running || pending > 0,
    enqueue,
    retry,
    clearFinished,
    discardPending,
    stopAfterCurrent: () => setPaused(true),
    resume: () => { if (!discardingRef.current) setPaused(false); },
  };
}
