import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import type {
  Bootstrap,
  Config,
  InputImage,
  LoraCandidate,
  RenderRequest,
  SavedImage,
  Workspace,
} from "./domain";

export interface PreprocessedImage {
  data_url: string;
  width: number;
  height: number;
}

export interface NativeBridge {
  bootstrap(): Promise<Bootstrap>;
  saveConfig(config: Config): Promise<Config>;
  selectWorkspace(path: string): Promise<Workspace>;
  readImage(path: string): Promise<InputImage>;
  listLoras(): Promise<LoraCandidate[]>;
  checkServer(config?: Config): Promise<unknown>;
  preprocess(image: string, kind: "canny" | "depth" | "pose"): Promise<PreprocessedImage>;
  render(sessionId: string, request: RenderRequest, context: object): Promise<SavedImage[]>;
  listImages(): Promise<SavedImage[]>;
  reveal(path: string): Promise<void>;
  chooseDirectory(title: string): Promise<string | null>;
  chooseImage(title: string): Promise<string | null>;
}

const nativeBridge: NativeBridge = {
  bootstrap: () => invoke<Bootstrap>("bootstrap"),
  saveConfig: (config) => invoke<Config>("save_config", { config }),
  selectWorkspace: (path) => invoke<Workspace>("select_workspace", { path }),
  readImage: (path) => invoke<InputImage>("read_image", { path }),
  listLoras: () => invoke<LoraCandidate[]>("list_loras"),
  checkServer: (config) => invoke<unknown>("check_server", { config: config ?? null }),
  preprocess: (image, kind) => invoke<PreprocessedImage>("preprocess", { image, kind }),
  render: (sessionId, request, context) => invoke<SavedImage[]>("render", { sessionId, request, context }),
  listImages: () => invoke<SavedImage[]>("list_images"),
  reveal: (path) => invoke<void>("reveal", { path }),
  chooseDirectory: (title) => open({ directory: true, multiple: false, title }),
  chooseImage: (title) => open({
    directory: false,
    multiple: false,
    title,
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg"] }],
  }),
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
  let cancelPicker = query.has("cancelPicker");
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
  const previewInput = (name: string): InputImage => ({ name, data_url: svgDataUrl(name, "#718074"), width: portraitInputs ? 768 : 1024, height: 1024 });
  return {
    bootstrap: async () => ({ config: { ...config }, config_path: "/Users/demo/.config/xwen/image-studio.json", workspace: firstLaunch ? null : { ...workspace } }),
    saveConfig: async (next) => (config = { ...next }),
    selectWorkspace: async (path) => {
      workspace = { path, session_id: `preview-session-${Date.now()}`, session_path: `${path}/preview-session` };
      config = { ...config, last_workspace: path, workspaces: [path, ...config.workspaces.filter((item) => item !== path)] };
      images = [];
      return { ...workspace };
    },
    readImage: async (path) => previewInput(path.split("/").pop() || "image.png"),
    listLoras: async () => [
      { name: "Ceramic light", path: "/models/ceramic-light.safetensors", size_bytes: 184_000_000 },
      { name: "Ink contour", path: "/models/ink-contour.safetensors", size_bytes: 96_000_000 },
    ],
    checkServer: async () => ({ status: "ok", service: "xwen" }),
    preprocess: async (_image, kind) => {
      await new Promise((resolve) => setTimeout(resolve, 100));
      return { data_url: svgDataUrl(`${kind} preview`, "#4d6870"), width: 1024, height: 1024 };
    },
    render: async (sessionId, request, context) => {
      const previewWindow = globalThis as typeof globalThis & { __XWEN_PREVIEW_REQUESTS__?: RenderRequest[] };
      previewWindow.__XWEN_PREVIEW_REQUESTS__ = [...(previewWindow.__XWEN_PREVIEW_REQUESTS__ ?? []), request];
      await new Promise((resolve) => setTimeout(resolve, 120));
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
    listImages: async () => images.map((image) => ({ ...image })),
    reveal: async () => undefined,
    chooseDirectory: async () => {
      if (cancelPicker) { cancelPicker = false; return null; }
      return "/Users/demo/Pictures/new-workspace";
    },
    chooseImage: async (title) => `/Users/demo/Pictures/${title.toLowerCase().includes("mask") ? "mask.png" : "source.png"}`,
  };
}

export const bridge: NativeBridge = import.meta.env.DEV && import.meta.env.VITE_BRIDGE_MODE === "preview"
  ? createPreviewBridge()
  : nativeBridge;
