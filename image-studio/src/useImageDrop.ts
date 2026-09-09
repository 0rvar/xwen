import { useCallback, useEffect, useRef, useState } from "react";
import { bridge, type FileDropEvent } from "./bridge";
import type { InputImage } from "./domain";
import { logMessage, reportError } from "./logging";

export type ImageDropTarget = "source" | "control";

const MAX_IMAGE_BYTES = 100_000_000;
const IMAGE_EXTENSION = /\.(?:png|jpe?g)$/i;

function targetAt(x: number, y: number): ImageDropTarget | null {
  const target = document.elementFromPoint(x, y)?.closest<HTMLElement>("[data-image-drop]")?.dataset.imageDrop;
  return target === "source" || target === "control" ? target : null;
}

function readFile(file: Blob, mode: "data-url" | "array-buffer"): Promise<string | ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the dropped image."));
    reader.onload = () => {
      if (typeof reader.result === "string" || reader.result instanceof ArrayBuffer) resolve(reader.result);
      else reject(new Error("Could not read the dropped image."));
    };
    if (mode === "data-url") reader.readAsDataURL(file);
    else reader.readAsArrayBuffer(file);
  });
}

function decodeDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("The dropped file is not a valid PNG or JPEG image."));
    image.src = dataUrl;
  });
}

async function browserImage(file: File): Promise<InputImage> {
  if (file.size > MAX_IMAGE_BYTES) throw new Error("Choose an image no larger than 100 MB.");
  const signature = new Uint8Array(await readFile(file.slice(0, 12), "array-buffer") as ArrayBuffer);
  const png = signature.length >= 8 && signature[0] === 0x89 && signature[1] === 0x50 && signature[2] === 0x4e && signature[3] === 0x47 && signature[4] === 0x0d && signature[5] === 0x0a && signature[6] === 0x1a && signature[7] === 0x0a;
  const jpeg = signature.length >= 3 && signature[0] === 0xff && signature[1] === 0xd8 && signature[2] === 0xff;
  if (!png && !jpeg) throw new Error("Choose exactly one PNG or JPEG image.");
  const rawDataUrl = await readFile(file, "data-url") as string;
  const encoded = rawDataUrl.slice(rawDataUrl.indexOf(",") + 1);
  const dataUrl = `data:image/${png ? "png" : "jpeg"};base64,${encoded}`;
  const dimensions = await decodeDimensions(dataUrl);
  if (!dimensions.width || !dimensions.height || dimensions.width > 8192 || dimensions.height > 8192) throw new Error("Image dimensions must be between 1 and 8192 pixels per side.");
  return { name: file.name || "dropped-image", data_url: dataUrl, ...dimensions };
}

export function useImageDrop(onImage: (target: ImageDropTarget, image: InputImage) => void, onError: (message: string) => void) {
  const [activeTarget, setActiveTarget] = useState<ImageDropTarget | null>(null);
  const latest = useRef<Record<ImageDropTarget, number>>({ source: 0, control: 0 });
  const callbacks = useRef({ onImage, onError });
  callbacks.current = { onImage, onError };

  const handleError = useCallback((error: unknown) => {
    reportError("image-drop", error);
    callbacks.current.onError(error instanceof Error ? error.message : String(error));
  }, []);
  const accept = useCallback(async (target: ImageDropTarget, load: () => Promise<InputImage>) => {
    const token = ++latest.current[target];
    try {
      const image = await load();
      if (latest.current[target] === token) {
        logMessage("debug", "image-drop", "accepted image", { target, width: image.width, height: image.height });
        callbacks.current.onImage(target, image);
      }
    } catch (error) {
      if (latest.current[target] === token) handleError(error);
    }
  }, [handleError]);
  const importPath = useCallback((target: ImageDropTarget, path: string) => {
    if (!IMAGE_EXTENSION.test(path)) { handleError(new Error("Choose exactly one PNG or JPEG image.")); return; }
    void accept(target, () => bridge.readImage(path));
  }, [accept, handleError]);
  const cancel = useCallback((target: ImageDropTarget) => { latest.current[target] += 1; }, []);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    let nativePathsValid = false;
    const nativeEvent = (event: FileDropEvent) => {
      if (event.type === "leave") { nativePathsValid = false; setActiveTarget(null); return; }
      const target = targetAt(event.x, event.y);
      if (event.type === "enter") nativePathsValid = event.paths.length === 1 && IMAGE_EXTENSION.test(event.paths[0]!);
      if (event.type === "over") { setActiveTarget(nativePathsValid ? target : null); return; }
      if (event.type === "enter") { setActiveTarget(nativePathsValid ? target : null); return; }
      setActiveTarget(null);
      nativePathsValid = false;
      logMessage("debug", "image-drop", "native drop", { target, file_count: event.paths.length });
      if (!target) return;
      if (event.paths.length !== 1 || !IMAGE_EXTENSION.test(event.paths[0]!)) { handleError(new Error("Choose exactly one PNG or JPEG image.")); return; }
      importPath(target, event.paths[0]!);
    };
    void bridge.onFileDrop(nativeEvent).then((cleanup) => {
      if (disposed) cleanup();
      else { unlisten = cleanup; logMessage("debug", "image-drop", "native listener ready"); }
    }).catch((error: unknown) => {
      if (!disposed) { reportError("image-drop-listener", error); callbacks.current.onError("Native file drop is unavailable. You can still choose images with the picker."); }
    });

    const hasFiles = (event: DragEvent) => event.dataTransfer ? [...event.dataTransfer.types].includes("Files") : false;
    const hasUri = (event: DragEvent) => event.dataTransfer ? [...event.dataTransfer.types].includes("text/uri-list") : false;
    const drag = (event: DragEvent) => {
      if (!hasFiles(event) && !hasUri(event)) { setActiveTarget(null); return; }
      event.preventDefault();
      if (!hasFiles(event)) { setActiveTarget(null); if (event.dataTransfer) event.dataTransfer.dropEffect = "none"; return; }
      const target = targetAt(event.clientX, event.clientY);
      setActiveTarget(target);
      if (event.dataTransfer) event.dataTransfer.dropEffect = target ? "copy" : "none";
    };
    const leave = (event: DragEvent) => {
      if (!event.relatedTarget) setActiveTarget(null);
    };
    const drop = (event: DragEvent) => {
      if (!hasFiles(event) && !hasUri(event)) return;
      event.preventDefault();
      setActiveTarget(null);
      if (!hasFiles(event)) return;
      const target = targetAt(event.clientX, event.clientY);
      logMessage("debug", "image-drop", "browser drop", { target, file_count: event.dataTransfer?.files.length ?? 0 });
      if (!target) return;
      const files = [...(event.dataTransfer?.files ?? [])];
      if (files.length !== 1) { handleError(new Error("Choose exactly one PNG or JPEG image.")); return; }
      void accept(target, () => browserImage(files[0]!));
    };
    document.addEventListener("dragenter", drag);
    document.addEventListener("dragover", drag);
    document.addEventListener("dragleave", leave);
    document.addEventListener("drop", drop);
    return () => {
      disposed = true;
      unlisten?.();
      document.removeEventListener("dragenter", drag);
      document.removeEventListener("dragover", drag);
      document.removeEventListener("dragleave", leave);
      document.removeEventListener("drop", drop);
    };
  }, [accept, handleError, importPath]);

  return { activeTarget, importPath, cancel };
}
