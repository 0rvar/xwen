import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type {
  Bootstrap,
  Config,
  InputImage,
  LoraCandidate,
  RenderRequest,
  SavedImage,
  Workspace,
} from "./domain";
import type { BatchDraft, BatchHandle } from "./batchDraft";

export interface PreprocessedImage {
  data_url: string;
  width: number;
  height: number;
}

export interface SessionSummary { session_id: string; image_count: number }
export type FileDropEvent =
  | { type: "enter" | "drop"; paths: string[]; x: number; y: number }
  | { type: "over"; x: number; y: number }
  | { type: "leave" };

export interface NativeBridge {
  bootstrap(): Promise<Bootstrap>;
  saveConfig(config: Config): Promise<Config>;
  selectWorkspace(path: string): Promise<Workspace>;
  readImage(path: string): Promise<InputImage>;
  listLoras(): Promise<LoraCandidate[]>;
  checkServer(config?: Config): Promise<unknown>;
  preprocess(image: string, kind: "canny" | "depth" | "pose"): Promise<PreprocessedImage>;
  createBatch(sessionId: string, draft: BatchDraft): Promise<BatchHandle>;
  renderBatchJob(sessionId: string, batchId: string, jobId: string): Promise<SavedImage[]>;
  discardBatchJobs(sessionId: string, batchId: string, jobIds: string[]): Promise<void>;
  listImages(): Promise<SavedImage[]>;
  listSessions(): Promise<SessionSummary[]>;
  deleteImage(workspacePath: string, sessionId: string, imageId: string): Promise<void>;
  deleteSession(workspacePath: string, sessionId: string): Promise<Workspace>;
  generatePrompt(idea: string): Promise<string>;
  getLogPath(): Promise<string>;
  reveal(path: string): Promise<void>;
  chooseDirectory(title: string): Promise<string | null>;
  chooseImage(title: string): Promise<string | null>;
  onFileDrop(callback: (event: FileDropEvent) => void): Promise<() => void>;
}

const nativeBridge: NativeBridge = {
  bootstrap: () => invoke<Bootstrap>("bootstrap"),
  saveConfig: (config) => invoke<Config>("save_config", { config }),
  selectWorkspace: (path) => invoke<Workspace>("select_workspace", { path }),
  readImage: (path) => invoke<InputImage>("read_image", { path }),
  listLoras: () => invoke<LoraCandidate[]>("list_loras"),
  checkServer: (config) => invoke<unknown>("check_server", { config: config ?? null }),
  preprocess: (image, kind) => invoke<PreprocessedImage>("preprocess", { image, kind }),
  createBatch: (sessionId, draft) => invoke<BatchHandle>("create_batch", { sessionId, draft }),
  renderBatchJob: (sessionId, batchId, jobId) => invoke<SavedImage[]>("render_batch_job", { sessionId, batchId, jobId }),
  discardBatchJobs: (sessionId, batchId, jobIds) => invoke<void>("discard_batch_jobs", { sessionId, batchId, jobIds }),
  listImages: () => invoke<SavedImage[]>("list_images"),
  listSessions: () => invoke<SessionSummary[]>("list_sessions"),
  deleteImage: (workspacePath, sessionId, imageId) => invoke<void>("delete_image", { workspacePath, sessionId, imageId }),
  deleteSession: (workspacePath, sessionId) => invoke<Workspace>("delete_session", { workspacePath, sessionId }),
  generatePrompt: (idea) => invoke<string>("generate_prompt", { idea }),
  getLogPath: () => invoke<string>("get_log_path"),
  reveal: (path) => invoke<void>("reveal", { path }),
  chooseDirectory: (title) => open({ directory: true, multiple: false, title }),
  chooseImage: (title) => open({
    directory: false,
    multiple: false,
    title,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg"] }],
  }),
  onFileDrop: async (callback) => {
    const window = getCurrentWindow();
    let scale = await window.scaleFactor();
    const unlistenScale = await window.onScaleChanged(({ payload }) => { scale = payload.scaleFactor; });
    try {
      const unlistenDrop = await getCurrentWebview().onDragDropEvent(({ payload }) => {
        if (payload.type === "leave") { callback({ type: "leave" }); return; }
        const position = payload.position.toLogical(scale);
        if (payload.type === "over") callback({ type: "over", x: position.x, y: position.y });
        else callback({ type: payload.type, paths: payload.paths, x: position.x, y: position.y });
      });
      return () => { unlistenDrop(); unlistenScale(); };
    } catch (error) {
      unlistenScale();
      throw error;
    }
  },
};

function svgDataUrl(label: string, color = "#bd684d"): string {
  const safe = label.replace(/[<>&"']/g, "");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024"><defs><linearGradient id="g" x2="1" y2="1"><stop stop-color="#e8c7af"/><stop offset=".55" stop-color="${color}"/><stop offset="1" stop-color="#4a3732"/></linearGradient></defs><rect width="1024" height="1024" fill="url(#g)"/><circle cx="706" cy="300" r="190" fill="#f4e9d8" opacity=".36"/><path d="M0 780 Q250 590 500 760 T1024 680V1024H0Z" fill="#2f3030" opacity=".62"/><text x="64" y="94" fill="#fff" font-family="system-ui" font-size="34">${safe}</text></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function createPreviewBridge(): NativeBridge {
  const query = new URLSearchParams(location.search);
  const firstLaunch = query.has("firstLaunch");
  const portraitInputs = query.has("portrait");
  const renderDelay = Number(query.get("renderDelay") ?? 120);
  const readDelay = Number(query.get("readDelay") ?? 0);
  const batchDelay = Number(query.get("batchDelay") ?? 0);
  let cancelPicker = query.has("cancelPicker");
  let failDiscard = query.has("failDiscard");
  let config: Config = {
    server_url: firstLaunch ? "" : "http://127.0.0.1:5241",
    api_key: "",
    workspaces: firstLaunch ? [] : ["/Users/demo/Pictures/xwen-studio"],
    last_workspace: firstLaunch ? null : "/Users/demo/Pictures/xwen-studio",
  };
  let workspace: Workspace = {
    path: config.last_workspace ?? "/Users/demo/Pictures/xwen-studio",
    session_id: "20260908-142210-preview",
    session_path: `${config.last_workspace}/20260908-142210-preview`,
  };
  let counter = 2;
  let batchCounter = 0;
  let images: SavedImage[] = [
    {
      id: "preview-1",
      path: `${workspace.session_path}/0001.png`,
      metadata_path: `${workspace.session_path}/0001.yaml`,
      data_url: svgDataUrl("Terracotta study"),
      seed: 42817,
      width: 1024,
      height: 1024,
      session_id: workspace.session_id,
      prompt: "A quiet architectural study in warm terracotta, soft afternoon light",
      metadata: {
        request: { prompt: "A quiet architectural study in warm terracotta, soft afternoon light", width: 1024, height: 1024, steps: 8, seed: 42817, n: 1, loras: [], init_image: "inputs/source.png", strength: 0.6 },
        assets: { init_image: { path: "inputs/source.png", sha256: "preview", width: 1024, height: 1024 } },
        created_at: "2026-09-08T14:23:04Z",
        duration_ms: 22410,
        output: { start_step: 0 },
      },
    },
  ];
  const failedOnce = new Set<string>();
  const batches = new Map<string, { sessionId: string; draft: BatchDraft; jobIds: string[]; discarded: Set<string> }>();
  const previewInput = (name: string): InputImage => ({ name, data_url: svgDataUrl(name, "#718074"), width: portraitInputs ? 768 : 1024, height: 1024 });
  const hydrateRequest = (request: RenderRequest, inputs: Record<string, string>): RenderRequest => ({
    ...request,
    loras: request.loras.map((lora) => ({ ...lora })),
    ...(request.init_image ? { init_image: inputs[request.init_image] ?? request.init_image } : {}),
    ...(request.mask ? { mask: inputs[request.mask] ?? request.mask } : {}),
    ...(request.control ? { control: { ...request.control, image: inputs[request.control.image] ?? request.control.image } } : {}),
  });
  return {
    bootstrap: async () => ({ config: { ...config }, config_path: "/Users/demo/.config/xwen/image-studio.json", workspace: firstLaunch ? null : { ...workspace } }),
    saveConfig: async (next) => (config = { ...next }),
    selectWorkspace: async (path) => {
      workspace = { path, session_id: `preview-session-${Date.now()}`, session_path: `${path}/preview-session` };
      config = { ...config, last_workspace: path, workspaces: [path, ...config.workspaces.filter((item) => item !== path)] };
      images = [];
      return { ...workspace };
    },
    readImage: async (path) => {
      if (readDelay) await new Promise((resolve) => setTimeout(resolve, readDelay));
      return previewInput(path.split("/").pop() || "image.png");
    },
    listLoras: async () => [
      { name: "Ceramic light", path: "/models/ceramic-light.safetensors", size_bytes: 184_000_000 },
      { name: "Ink contour", path: "/models/ink-contour.safetensors", size_bytes: 96_000_000 },
    ],
    checkServer: async () => ({ status: "ok", service: "xwen" }),
    preprocess: async (_image, kind) => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      return { data_url: svgDataUrl(`${kind} preview`, "#4d6870"), width: 1024, height: 1024 };
    },
    createBatch: async (sessionId, draft) => {
      if (batchDelay) await new Promise((resolve) => setTimeout(resolve, batchDelay));
      if (draft.jobs.some((job) => job.request.prompt.toLowerCase().includes("fail manifest"))) throw new Error("Preview batch creation failed as requested.");
      const id = `preview-batch-${++batchCounter}`;
      const jobIds = draft.jobs.map((_, index) => `job-${String(index + 1).padStart(6, "0")}`);
      batches.set(id, { sessionId, draft, jobIds, discarded: new Set() });
      const handle = { id, manifest_path: `${workspace.session_path}/batches/${id}.yaml`, job_ids: jobIds };
      const previewWindow = globalThis as typeof globalThis & {
        __XWEN_PREVIEW_BATCH_DRAFTS__?: BatchDraft[];
        __XWEN_PREVIEW_BATCHES__?: Array<{ handle: BatchHandle; draft: BatchDraft }>;
      };
      previewWindow.__XWEN_PREVIEW_BATCH_DRAFTS__ = [...(previewWindow.__XWEN_PREVIEW_BATCH_DRAFTS__ ?? []), draft];
      previewWindow.__XWEN_PREVIEW_BATCHES__ = [...(previewWindow.__XWEN_PREVIEW_BATCHES__ ?? []), { handle, draft }];
      return handle;
    },
    renderBatchJob: async (sessionId, batchId, jobId) => {
      const batch = batches.get(batchId);
      if (!batch || batch.sessionId !== sessionId) throw new Error("Unknown preview batch.");
      const index = batch.jobIds.indexOf(jobId);
      if (index < 0 || batch.discarded.has(jobId)) throw new Error("Preview batch job is unavailable.");
      const planned = batch.draft.jobs[index]!;
      const request = hydrateRequest(planned.request, batch.draft.inputs);
      const context = { ...planned.context, batch_id: batchId, job_id: jobId };
      const previewWindow = globalThis as typeof globalThis & {
        __XWEN_PREVIEW_REQUESTS__?: RenderRequest[];
        __XWEN_PREVIEW_BATCH_RENDERS__?: Array<{ batchId: string; jobId: string }>;
      };
      previewWindow.__XWEN_PREVIEW_REQUESTS__ = [...(previewWindow.__XWEN_PREVIEW_REQUESTS__ ?? []), request];
      previewWindow.__XWEN_PREVIEW_BATCH_RENDERS__ = [...(previewWindow.__XWEN_PREVIEW_BATCH_RENDERS__ ?? []), { batchId, jobId }];
      await new Promise((resolve) => setTimeout(resolve, renderDelay));
      const failureKey = `${request.prompt}|${request.seed}`;
      if (request.prompt.toLowerCase().includes("fail preview") && !failedOnce.has(failureKey)) {
        failedOnce.add(failureKey);
        throw new Error("Preview render failed as requested.");
      }
      const id = `preview-${counter++}`;
      const saved: SavedImage = {
        id,
        path: `${workspace.session_path}/${id}.png`,
        metadata_path: `${workspace.session_path}/${id}.yaml`,
        data_url: svgDataUrl(request.prompt || "Generated image", counter % 2 ? "#b8654c" : "#826252"),
        seed: request.seed,
        width: request.width,
        height: request.height,
        session_id: sessionId,
        prompt: request.prompt,
        metadata: { request, context, created_at: new Date().toISOString(), duration_ms: 22140, output: { start_step: request.init_image ? 3 : 0 } },
      };
      images = [saved, ...images];
      return [saved];
    },
    discardBatchJobs: async (sessionId, batchId, jobIds) => {
      const batch = batches.get(batchId);
      if (!batch || batch.sessionId !== sessionId || jobIds.some((id) => !batch.jobIds.includes(id))) throw new Error("Unknown preview batch jobs.");
      if (failDiscard) { failDiscard = false; throw new Error("Preview discard failed as requested."); }
      jobIds.forEach((id) => batch.discarded.add(id));
      const previewWindow = globalThis as typeof globalThis & { __XWEN_PREVIEW_DISCARDS__?: Array<{ batchId: string; jobIds: string[] }> };
      previewWindow.__XWEN_PREVIEW_DISCARDS__ = [...(previewWindow.__XWEN_PREVIEW_DISCARDS__ ?? []), { batchId, jobIds: [...jobIds] }];
    },
    listImages: async () => images.map((image) => ({ ...image })),
    listSessions: async () => [...new Set([workspace.session_id, ...images.map((image) => image.session_id)])].map((session_id) => ({ session_id, image_count: images.filter((image) => image.session_id === session_id).length })),
    deleteImage: async (_path, sessionId, imageId) => { images = images.filter((image) => image.session_id !== sessionId || image.id !== imageId); },
    deleteSession: async (_path, sessionId) => {
      images = images.filter((image) => image.session_id !== sessionId);
      if (workspace.session_id === sessionId) {
        const session_id = `preview-session-${Date.now()}`;
        workspace = { ...workspace, session_id, session_path: `${workspace.path}/${session_id}` };
      }
      return { ...workspace };
    },
    generatePrompt: async (idea) => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      if (idea.includes("fail prompt")) throw new Error("Flash-Next is not cached on this server.");
      return `${idea.trim() || "A secluded mountain observatory"}, soft evening light, rich textures, carefully composed photograph`;
    },
    getLogPath: async () => "/Users/demo/.local/state/xwen/image-studio/logs/image-studio.log",
    reveal: async () => undefined,
    chooseDirectory: async () => {
      if (cancelPicker) { cancelPicker = false; return null; }
      return "/Users/demo/Pictures/new-workspace";
    },
    chooseImage: async (title) => `/Users/demo/Pictures/${title.toLowerCase().includes("mask") ? "mask.png" : "source.png"}`,
    onFileDrop: async () => () => undefined,
  };
}

export const bridge: NativeBridge = import.meta.env.DEV && import.meta.env.VITE_BRIDGE_MODE === "preview"
  ? createPreviewBridge()
  : nativeBridge;
