/// <reference lib="webworker" />
import { runPipelineRGBA, type PipelineOutput } from "./dip";

self.onmessage = (e: MessageEvent<{ rgba: ArrayBuffer; w: number; h: number }>) => {
  const { rgba, w, h } = e.data;
  const out = runPipelineRGBA(new Uint8ClampedArray(rgba), w, h);
  // Transfer all the buffers back.
  const transfer: Transferable[] = [
    out.grayscale.buffer as ArrayBuffer,
    out.equalized.buffer as ArrayBuffer,
    out.lungMask.buffer as ArrayBuffer,
    out.saliency.buffer as ArrayBuffer,
    out.overlayRGBA.buffer as ArrayBuffer,
    out.heatRGBA.buffer as ArrayBuffer,
  ];
  (self as unknown as Worker).postMessage(out as PipelineOutput, transfer);
};

export {};
