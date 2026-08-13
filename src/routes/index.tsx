import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Upload, Loader2, AlertTriangle, ScanLine, Sparkles, Flame } from "lucide-react";
import { CanvasView } from "@/components/CanvasView";
import { MetricBar } from "@/components/MetricBar";
import type { PipelineOutput, RegionFeatures } from "@/lib/dip";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "LungTumorVision — CT Tumor Detection" },
      {
        name: "description",
        content:
          "Upload a lung CT slice. CLAHE histogram equalization, lung segmentation, and a Grad-CAM-style saliency map highlight suspicious regions, then AI scores tumor likelihood.",
      },
    ],
  }),
});

interface AIResult {
  tumorLikelihood: number;
  isLungCT: boolean;
  classification: "normal" | "benign" | "malignant" | "indeterminate";
  severity: string;
  rationale: string;
  recommendations: string[];
}

const MAX_DIM = 384; // smaller -> faster, still good for DIP

async function fileToImageData(
  file: File,
): Promise<{ rgba: Uint8ClampedArray; w: number; h: number; dataUrl: string }> {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = url;
    });
    const scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    return { rgba: data, w, h, dataUrl: c.toDataURL("image/jpeg", 0.85) };
  } finally {
    URL.revokeObjectURL(url);
  }
}

// helper: build ImageData from single-channel array
function makeGrayImageData(gray: Uint8ClampedArray, w: number, h: number): ImageData {
  const buf = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, j = 0; j < gray.length; i += 4, j++) {
    buf[i] = buf[i + 1] = buf[i + 2] = gray[j];
    buf[i + 3] = 255;
  }
  return new ImageData(buf, w, h);
}
function makeRGBAImageData(rgba: Uint8ClampedArray, w: number, h: number): ImageData {
  return new ImageData(new Uint8ClampedArray(rgba), w, h);
}

function cropDataUrl(
  rgba: Uint8ClampedArray, w: number, h: number,
  bbox: { x: number; y: number; w: number; h: number },
): string {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  const out = document.createElement("canvas");
  out.width = bbox.w; out.height = bbox.h;
  out.getContext("2d")!.drawImage(c, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, bbox.w, bbox.h);
  return out.toDataURL("image/jpeg", 0.85);
}

function Index() {
  const inputRef = useRef<HTMLInputElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const [output, setOutput] = useState<PipelineOutput | null>(null);
  const [imgData, setImgData] = useState<{ rgba: Uint8ClampedArray; w: number; h: number; dataUrl: string } | null>(null);
  const [ai, setAi] = useState<AIResult | null>(null);
  const [stage, setStage] = useState<"idle" | "processing" | "ai" | "done">("idle");
  const [fileName, setFileName] = useState<string>("");

  // Long-lived worker.
  useEffect(() => {
    const w = new Worker(new URL("../lib/dip.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = w;
    return () => { w.terminate(); workerRef.current = null; };
  }, []);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload a CT scan image (PNG, JPG).");
      return;
    }
    setFileName(file.name);
    setAi(null);
    setStage("processing");
    setOutput(null);

    try {
      const img = await fileToImageData(file);
      setImgData(img);

      // Run DIP off main thread.
      const result = await new Promise<PipelineOutput>((resolve, reject) => {
        const w = workerRef.current;
        if (!w) return reject(new Error("Worker not ready"));
        const onMsg = (e: MessageEvent<PipelineOutput>) => {
          w.removeEventListener("message", onMsg);
          resolve(e.data);
        };
        w.addEventListener("message", onMsg);
        // Clone rgba so we still have it on main thread.
        const copy = new Uint8ClampedArray(img.rgba);
        w.postMessage({ rgba: copy.buffer, w: img.w, h: img.h }, [copy.buffer]);
      });
      setOutput(result);

      if (!result.region || !result.regionBBoxPad) {
        setStage("done");
        return;
      }

      setStage("ai");
      const regionCrop = cropDataUrl(img.rgba, img.w, img.h, result.regionBBoxPad);
      // Also send a heat-map crop centered on the ROI for extra context.
      const heatCrop = cropDataUrl(result.heatRGBA, img.w, img.h, result.regionBBoxPad);

      const { data, error } = await supabase.functions.invoke("analyze-scan", {
        body: {
          imageBase64: img.dataUrl,
          regionBase64: regionCrop,
          heatBase64: heatCrop,
          features: {
            area: result.region.area,
            areaRatio: result.region.areaRatio,
            meanIntensity: result.region.meanIntensity,
            stdIntensity: result.region.stdIntensity,
            circularity: result.region.circularity,
            solidity: result.region.solidity,
            edgeIrregularity: result.region.edgeIrregularity,
            contrast: result.region.contrast,
            saliencyMean: result.region.saliencyMean,
          },
        },
      });
      if (error) throw error;
      if (data?.error) throw new Error(data.error);
      setAi(data as AIResult);
      setStage("done");
    } catch (e) {
      console.error(e);
      toast.error(e instanceof Error ? e.message : "Analysis failed");
      setStage("done");
    }
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  }, [handleFile]);

  // Memoize ImageData conversions so re-renders don't re-allocate.
  const views = useMemo(() => {
    if (!output) return null;
    const { width: w, height: h } = output;
    return {
      grayscale: makeGrayImageData(output.grayscale, w, h),
      equalized: makeGrayImageData(output.equalized, w, h),
      lungMask: makeGrayImageData(output.lungMask, w, h),
      saliency: makeGrayImageData(output.saliency, w, h),
      heat: makeRGBAImageData(output.heatRGBA, w, h),
      overlay: makeRGBAImageData(output.overlayRGBA, w, h),
    };
  }, [output]);

  const tumorPct = ai?.tumorLikelihood ?? 0;
  const tumorTone =
    tumorPct < 0.3 ? "success" : tumorPct < 0.6 ? "warning" : "destructive";
  const growthTone =
    !output ? "primary"
      : output.growthScore < 0.3 ? "success"
      : output.growthScore < 0.6 ? "warning"
      : "destructive";
  const classBadge: Record<string, { label: string; tone: string }> = {
    normal: { label: "NORMAL", tone: "var(--success)" },
    benign: { label: "BENIGN", tone: "var(--warning)" },
    malignant: { label: "MALIGNANT", tone: "var(--destructive)" },
    indeterminate: { label: "INDETERMINATE", tone: "var(--muted-foreground)" },
  };

  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <header className="max-w-7xl mx-auto flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="size-10 rounded-md bg-primary/15 border border-primary/30 grid place-items-center glow-ring">
            <Activity className="size-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">LungTumorVision</h1>
            <p className="text-[11px] font-mono uppercase tracking-[0.25em] text-muted-foreground">
              CT · CLAHE + saliency + AI
            </p>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-muted-foreground border border-border rounded-full px-3 py-1.5">
          <span className="size-1.5 rounded-full bg-success animate-pulse" />
          system online
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid lg:grid-cols-[1fr_380px] gap-6">
        <section className="space-y-6">
          {!output && (
            <div
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => inputRef.current?.click()}
              className="relative aspect-[16/10] rounded-xl border-2 border-dashed border-border hover:border-primary/60 bg-card/50 grid place-items-center cursor-pointer transition-all group"
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <div className="text-center">
                <div className="mx-auto size-16 rounded-full bg-primary/10 border border-primary/30 grid place-items-center mb-4 group-hover:scale-110 transition-transform">
                  {stage === "processing" || stage === "ai" ? (
                    <Loader2 className="size-7 text-primary animate-spin" />
                  ) : (
                    <Upload className="size-7 text-primary" />
                  )}
                </div>
                <p className="text-lg font-medium">Drop a lung CT slice</p>
                <p className="text-sm text-muted-foreground mt-1">
                  PNG · JPG · processed locally in a Web Worker
                </p>
                <p className="text-[10px] font-mono uppercase tracking-[0.25em] text-muted-foreground/60 mt-6">
                  Educational demo — not a medical device
                </p>
              </div>
            </div>
          )}

          {output && views && (
            <>
              {/* Pipeline strip */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <CanvasView data={views.grayscale} label="01 · Grayscale" hint="Y' luma" />
                <CanvasView data={views.equalized} label="02 · CLAHE" hint="adaptive hist eq" />
                <CanvasView data={views.lungMask} label="03 · Lung mask" hint="Otsu + morphology" />
                <CanvasView data={views.saliency} label="04 · Saliency" hint="grad + entropy + dev" />
                <CanvasView data={views.heat} label="05 · Heat map" hint="Grad-CAM-style" />
                <CanvasView data={views.overlay} label="06 · Detected ROI" hint={output.region ? "candidate" : "none"} />
              </div>

              <div className="rounded-xl border border-border bg-card/60 backdrop-blur p-5 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <ScanLine className="size-5 text-primary" />
                  <div>
                    <p className="text-sm font-medium">{fileName || "scan.png"}</p>
                    <p className="text-[11px] font-mono text-muted-foreground">
                      {output.width}×{output.height}px ·{" "}
                      {output.region ? "1 ROI" : "no ROI"} ·{" "}
                      {stage === "ai" ? "querying AI…" : "done"}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setOutput(null); setAi(null); setStage("idle"); setImgData(null);
                  }}
                  className="text-xs font-mono uppercase tracking-[0.2em] px-4 py-2 rounded-md border border-border hover:border-primary/60 hover:text-primary transition-colors"
                >
                  New scan
                </button>
              </div>
            </>
          )}
        </section>

        <aside className="space-y-5">
          <div className="rounded-xl border border-border bg-card/60 backdrop-blur p-5">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="size-4 text-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em]">AI Confidence</h2>
            </div>

            {stage === "idle" && (
              <p className="text-sm text-muted-foreground">Awaiting scan upload.</p>
            )}

            {(stage === "processing" || stage === "ai") && (
              <div className="flex items-center gap-3 py-4 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin text-primary" />
                {stage === "processing" ? "Running DIP pipeline…" : "Querying AI vision model…"}
              </div>
            )}

            {stage === "done" && ai && (
              <div className="space-y-5">
                {!ai.isLungCT && (
                  <div className="flex gap-2 text-xs p-3 rounded-md bg-warning/10 border border-warning/30 text-warning">
                    <AlertTriangle className="size-4 shrink-0 mt-0.5" />
                    Image does not appear to be a lung CT — results unreliable.
                  </div>
                )}
                {ai.classification && (
                  <div>
                    <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-1.5">
                      Classification
                    </p>
                    <div
                      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md font-mono text-xs tracking-wider"
                      style={{
                        background: `color-mix(in oklab, ${classBadge[ai.classification]?.tone} 15%, transparent)`,
                        border: `1px solid color-mix(in oklab, ${classBadge[ai.classification]?.tone} 50%, transparent)`,
                        color: classBadge[ai.classification]?.tone,
                      }}
                    >
                      <Flame className="size-3" />
                      {classBadge[ai.classification]?.label ?? ai.classification.toUpperCase()}
                    </div>
                  </div>
                )}
                <MetricBar
                  label="Tumor likelihood"
                  value={ai.tumorLikelihood}
                  display={`${(ai.tumorLikelihood * 100).toFixed(1)}%`}
                  tone={tumorTone}
                />
                <MetricBar
                  label="Growth indicators (heuristic)"
                  value={output?.growthScore ?? 0}
                  display={output?.growthLabel}
                  tone={growthTone}
                />
                <div>
                  <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-1.5">
                    Severity
                  </p>
                  <p className="text-sm">{ai.severity}</p>
                </div>
                <div>
                  <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-1.5">
                    Rationale
                  </p>
                  <p className="text-sm leading-relaxed text-foreground/80">{ai.rationale}</p>
                </div>
                {ai.recommendations?.length > 0 && (
                  <div>
                    <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-1.5">
                      Next steps
                    </p>
                    <ul className="space-y-1.5">
                      {ai.recommendations.map((r, i) => (
                        <li key={i} className="text-sm flex gap-2">
                          <span className="text-primary">›</span>
                          <span className="text-foreground/80">{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {stage === "done" && !ai && output && !output.region && (
              <p className="text-sm text-muted-foreground">
                No suspicious region was detected by the DIP pipeline. The scan appears clear or is not a lung CT.
              </p>
            )}
          </div>

          {output?.region && (
            <div className="rounded-xl border border-border bg-card/60 backdrop-blur p-5">
              <h3 className="text-xs font-mono uppercase tracking-[0.2em] text-muted-foreground mb-4">
                Region features
              </h3>
              <Features f={output.region} />
            </div>
          )}

          <p className="text-[10px] font-mono leading-relaxed text-muted-foreground/60 px-1">
            Disclaimer · Combines classical DIP (CLAHE, Otsu, morphology, lung segmentation,
            saliency / Grad-CAM-style heat map, connected components) with an AI vision model
            for educational demonstration only. NOT a diagnostic device.
          </p>
        </aside>
      </div>
    </main>
  );
}

function Features({ f }: { f: RegionFeatures }) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm font-mono tabular-nums">
      <Stat k="area" v={`${f.area} px`} />
      <Stat k="area %" v={`${(f.areaRatio * 100).toFixed(2)}%`} />
      <Stat k="μ intensity" v={f.meanIntensity.toFixed(1)} />
      <Stat k="σ intensity" v={f.stdIntensity.toFixed(1)} />
      <Stat k="circularity" v={f.circularity.toFixed(3)} />
      <Stat k="solidity" v={f.solidity.toFixed(3)} />
      <Stat k="edge irreg." v={f.edgeIrregularity.toFixed(3)} />
      <Stat k="contrast" v={f.contrast.toFixed(3)} />
      <Stat k="saliency μ" v={f.saliencyMean.toFixed(3)} />
    </dl>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{k}</dt>
      <dd className="text-foreground text-right">{v}</dd>
    </>
  );
}
