import { useCallback, useEffect, useRef, useState } from "react";
import type { InputImage } from "../domain";
import { bridge } from "../bridge";

interface Props {
  source: InputImage;
  initial: InputImage | null;
  onCancel(): void;
  onSave(mask: InputImage): void;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not decode the image."));
    image.src = src;
  });
}

export function MaskEditor({ source, initial, onCancel, onSave }: Props) {
  const displayRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const sourceRef = useRef<HTMLImageElement | null>(null);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const [tool, setTool] = useState<"paint" | "erase">("paint");
  const [brush, setBrush] = useState(64);
  const [error, setError] = useState("");

  const redraw = useCallback(() => {
    const display = displayRef.current;
    const mask = maskRef.current;
    const image = sourceRef.current;
    if (!display || !mask || !image) return;
    const ctx = display.getContext("2d", { willReadFrequently: true });
    const maskCtx = mask.getContext("2d", { willReadFrequently: true });
    if (!ctx || !maskCtx) return;
    ctx.clearRect(0, 0, source.width, source.height);
    ctx.drawImage(image, 0, 0, source.width, source.height);
    const pixels = maskCtx.getImageData(0, 0, source.width, source.height).data;
    const overlay = ctx.getImageData(0, 0, source.width, source.height);
    for (let i = 0; i < pixels.length; i += 4) {
      const alpha = pixels[i]! / 255 * 0.48;
      if (alpha <= 0) continue;
      overlay.data[i] = Math.round(221 * alpha + overlay.data[i]! * (1 - alpha));
      overlay.data[i + 1] = Math.round(72 * alpha + overlay.data[i + 1]! * (1 - alpha));
      overlay.data[i + 2] = Math.round(52 * alpha + overlay.data[i + 2]! * (1 - alpha));
    }
    ctx.putImageData(overlay, 0, 0);
  }, [source.height, source.width]);

  useEffect(() => {
    let disposed = false;
    void Promise.all([loadImage(source.data_url), initial ? loadImage(initial.data_url) : Promise.resolve(null)]).then(([sourceImage, maskImage]) => {
      if (disposed) return;
      sourceRef.current = sourceImage;
      const mask = document.createElement("canvas");
      mask.width = source.width;
      mask.height = source.height;
      const ctx = mask.getContext("2d", { willReadFrequently: true })!;
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, mask.width, mask.height);
      if (maskImage) ctx.drawImage(maskImage, 0, 0, mask.width, mask.height);
      maskRef.current = mask;
      redraw();
    }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
    return () => { disposed = true; };
  }, [initial, redraw, source]);

  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * source.width / rect.width,
      y: (event.clientY - rect.top) * source.height / rect.height,
    };
  };
  const draw = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const ctx = maskRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = tool === "paint" ? "white" : "black";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    redraw();
  };
  const mutate = (kind: "clear" | "invert") => {
    const canvas = maskRef.current;
    const ctx = canvas?.getContext("2d", { willReadFrequently: true });
    if (!canvas || !ctx) return;
    if (kind === "clear") {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
      const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < image.data.length; i += 4) {
        const value = 255 - image.data[i]!;
        image.data[i] = value;
        image.data[i + 1] = value;
        image.data[i + 2] = value;
        image.data[i + 3] = 255;
      }
      ctx.putImageData(image, 0, 0);
    }
    redraw();
  };
  const importMask = async () => {
    const path = await bridge.chooseImage("Import a black and white mask");
    if (!path) return;
    const selected = await bridge.readImage(path);
    const image = await loadImage(selected.data_url);
    const canvas = maskRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    redraw();
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal mask-modal" role="dialog" aria-modal="true" aria-labelledby="mask-title">
        <header className="modal-heading">
          <div><p className="eyebrow">Inpaint mask</p><h2 id="mask-title">Mark areas to repaint</h2></div>
          <button className="icon-button" onClick={onCancel} aria-label="Close mask editor">×</button>
        </header>
        <p className="muted">Painted red areas become white in the saved mask. Black areas keep the source.</p>
        <div className="mask-toolbar" aria-label="Mask tools">
          <div className="segmented compact">
            <button className={tool === "paint" ? "selected" : ""} onClick={() => setTool("paint")}>Paint</button>
            <button className={tool === "erase" ? "selected" : ""} onClick={() => setTool("erase")}>Erase</button>
          </div>
          <label className="inline-field">Brush <input aria-label="Brush size" type="range" min="8" max="280" value={brush} onChange={(event) => setBrush(Number(event.target.value))} /> <span>{brush}px</span></label>
          <button className="quiet-button" onClick={() => mutate("clear")}>Clear</button>
          <button className="quiet-button" onClick={() => mutate("invert")}>Invert</button>
          <button className="quiet-button" onClick={() => void importMask().catch((reason: unknown) => setError(String(reason)))}>Import</button>
        </div>
        {error && <p className="error-banner" role="alert">{error}</p>}
        <div className="mask-stage">
          <canvas
            ref={displayRef}
            width={source.width}
            height={source.height}
            aria-label="Mask painting canvas"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              const at = point(event);
              lastRef.current = at;
              draw(at, at);
            }}
            onPointerMove={(event) => {
              if (!event.currentTarget.hasPointerCapture(event.pointerId) || !lastRef.current) return;
              const at = point(event);
              draw(lastRef.current, at);
              lastRef.current = at;
            }}
            onPointerUp={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
              lastRef.current = null;
            }}
            onPointerCancel={() => { lastRef.current = null; }}
          />
        </div>
        <footer className="modal-actions">
          <button className="quiet-button" onClick={onCancel}>Cancel</button>
          <button className="primary-button" onClick={() => {
            const canvas = maskRef.current;
            if (!canvas) return;
            onSave({ name: `mask-${source.name.replace(/\.[^.]+$/, "")}.png`, data_url: canvas.toDataURL("image/png"), width: source.width, height: source.height });
          }}>Use mask</button>
        </footer>
      </section>
    </div>
  );
}
