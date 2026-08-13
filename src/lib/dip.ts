// Digital image processing pipeline for lung CT slice analysis.
// Pure functions — runnable in a Web Worker (no DOM access).

// Pipeline:
//   1. Grayscale conversion
//   2. CLAHE-style tile histogram equalization
//   3. Lung-field segmentation (Otsu on inverted image + morphology)
//   4. Saliency map ("Grad-CAM-like" via local entropy + gradient + intensity deviation)
//   5. Otsu threshold on saliency, restricted to lung mask
//   6. Connected components -> candidates with shape/intensity features

export interface RegionFeatures {
  area: number;
  areaRatio: number;       // candidate / lung-mask area
  meanIntensity: number;
  stdIntensity: number;
  circularity: number;     // 4πA / P²
  solidity: number;        // area / convex-hull area
  edgeIrregularity: number;
  contrast: number;        // mean(inside) vs mean(ring outside)
  saliencyMean: number;    // 0..1
  bbox: { x: number; y: number; w: number; h: number };
  centroid: { x: number; y: number };
  perimeter: number;
}

export interface PipelineOutput {
  width: number;
  height: number;
  grayscale: Uint8ClampedArray;   // single-channel buffers
  equalized: Uint8ClampedArray;
  lungMask: Uint8ClampedArray;    // 0/255
  saliency: Uint8ClampedArray;    // heat 0..255
  // Pre-rendered RGBA for direct putImageData on main thread:
  overlayRGBA: Uint8ClampedArray; // original w/ ROI outlined + heat overlay
  heatRGBA: Uint8ClampedArray;    // pseudo-color jet heat map blended on equalized
  region?: RegionFeatures;
  regionBBoxPad: { x: number; y: number; w: number; h: number } | null;
  growthScore: number;
  growthLabel: string;
}

// -------------------- helpers --------------------

export function rgbaToGray(rgba: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba.length / 4);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    out[j] = (rgba[i] * 0.299 + rgba[i + 1] * 0.587 + rgba[i + 2] * 0.114) | 0;
  }
  return out;
}

// CLAHE-style: split into tiles, equalize each tile's histogram, bilinearly
// interpolate per-pixel between tile mappings.
function clahe(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  tilesX = 8,
  tilesY = 8,
  clipLimit = 0.01, // fraction of pixels per bin
): Uint8ClampedArray {
  const tw = Math.ceil(w / tilesX);
  const th = Math.ceil(h / tilesY);
  // For each tile, build clipped CDF lookup [256].
  const luts = new Array<Uint8Array>(tilesX * tilesY);
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * tw, y0 = ty * th;
      const x1 = Math.min(w, x0 + tw), y1 = Math.min(h, y0 + th);
      const hist = new Uint32Array(256);
      let n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) { hist[gray[y * w + x]]++; n++; }
      }
      // Clip and redistribute.
      const limit = Math.max(1, Math.floor(clipLimit * n));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; }
      }
      const add = Math.floor(excess / 256);
      let leftover = excess - add * 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += add;
        if (leftover > 0) { hist[i]++; leftover--; }
      }
      // CDF.
      const lut = new Uint8Array(256);
      let acc = 0;
      const total = n || 1;
      for (let i = 0; i < 256; i++) {
        acc += hist[i];
        lut[i] = Math.round((acc * 255) / total);
      }
      luts[ty * tilesX + tx] = lut;
    }
  }
  // Bilinear interpolate between 4 tile centers.
  const out = new Uint8ClampedArray(gray.length);
  for (let y = 0; y < h; y++) {
    const fy = (y + 0.5) / th - 0.5;
    const ty0 = Math.max(0, Math.min(tilesY - 1, Math.floor(fy)));
    const ty1 = Math.max(0, Math.min(tilesY - 1, ty0 + 1));
    const wy = fy - ty0;
    for (let x = 0; x < w; x++) {
      const fx = (x + 0.5) / tw - 0.5;
      const tx0 = Math.max(0, Math.min(tilesX - 1, Math.floor(fx)));
      const tx1 = Math.max(0, Math.min(tilesX - 1, tx0 + 1));
      const wx = fx - tx0;
      const v = gray[y * w + x];
      const a = luts[ty0 * tilesX + tx0][v];
      const b = luts[ty0 * tilesX + tx1][v];
      const c = luts[ty1 * tilesX + tx0][v];
      const d = luts[ty1 * tilesX + tx1][v];
      out[y * w + x] =
        (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy) | 0;
    }
  }
  return out;
}

function otsu(gray: Uint8ClampedArray, mask?: Uint8Array): number {
  const hist = new Uint32Array(256);
  let total = 0;
  if (mask) {
    for (let i = 0; i < gray.length; i++) if (mask[i]) { hist[gray[i]]++; total++; }
  } else {
    for (const v of gray) hist[v]++;
    total = gray.length;
  }
  if (!total) return 127;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, max = 0, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > max) { max = between; threshold = t; }
  }
  return threshold;
}

function morph(bin: Uint8Array, w: number, h: number, op: "erode" | "dilate", iter = 1): Uint8Array {
  let cur = bin;
  for (let k = 0; k < iter; k++) {
    const out = new Uint8Array(cur.length);
    const target = op === "erode" ? 1 : 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        let allMatch = true;
        outer: for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (cur[i + dy * w + dx] !== target) { allMatch = false; break outer; }
          }
        }
        if (op === "erode") out[i] = allMatch ? 1 : 0;
        else out[i] = allMatch ? 0 : 1;
      }
    }
    cur = out;
  }
  return cur;
}

// Largest non-background connected components → fill = lung mask.
function lungMask(gray: Uint8ClampedArray, w: number, h: number): Uint8Array {
  // Lungs are dark. Threshold inverted image with Otsu.
  const inv = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i++) inv[i] = 255 - gray[i];
  const t = otsu(inv);
  const bin = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) bin[i] = inv[i] >= t ? 1 : 0;
  // Remove border-touching = body outside? Actually lungs are interior dark blobs.
  // Open then close.
  let m = morph(bin, w, h, "erode", 1);
  m = morph(m, w, h, "dilate", 2);
  // Keep connected components that are NOT touching the border (lungs are interior),
  // and have reasonable size.
  const labels = new Int32Array(m.length);
  const out = new Uint8Array(m.length);
  let next = 1;
  const stack: number[] = [];
  const minA = w * h * 0.01;
  for (let i = 0; i < m.length; i++) {
    if (m[i] !== 1 || labels[i]) continue;
    stack.push(i);
    labels[i] = next;
    const pixels: number[] = [];
    let touchesBorder = false;
    while (stack.length) {
      const p = stack.pop()!;
      pixels.push(p);
      const px = p % w, py = (p / w) | 0;
      if (px === 0 || py === 0 || px === w - 1 || py === h - 1) touchesBorder = true;
      const nbrs = [p - 1, p + 1, p - w, p + w];
      for (const n of nbrs) {
        if (n < 0 || n >= m.length) continue;
        if (m[n] === 1 && !labels[n]) { labels[n] = next; stack.push(n); }
      }
    }
    if (!touchesBorder && pixels.length > minA) {
      for (const px of pixels) out[px] = 1;
    }
    next++;
  }
  // If we found nothing (image is not a chest CT), fall back: use full frame minus 5% border.
  let any = 0;
  for (let i = 0; i < out.length; i++) any += out[i];
  if (any < w * h * 0.02) {
    out.fill(0);
    const pad = Math.round(Math.min(w, h) * 0.05);
    for (let y = pad; y < h - pad; y++)
      for (let x = pad; x < w - pad; x++) out[y * w + x] = 1;
  }
  // Dilate slightly to recover lesions on lung wall.
  return morph(out, w, h, "dilate", 2);
}

// -------------------- saliency (Grad-CAM-style heuristic) --------------------
//
// Combines:
//  - Sobel gradient magnitude (edge energy)
//  - Local intensity deviation from neighborhood mean (anomaly)
//  - Local entropy of intensities (texture)
// All within the lung mask. Output normalized to 0..255.

function sobel(gray: Uint8ClampedArray, w: number, h: number): Float32Array {
  const out = new Float32Array(gray.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] + gray[i - w + 1]
        - 2 * gray[i - 1] + 2 * gray[i + 1]
        - gray[i + w - 1] + gray[i + w + 1];
      const gy =
        -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1]
        + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
      out[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return out;
}

// Box blur (mean filter) using separable passes — O(N).
function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const win = r * 2 + 1;
  // Horizontal
  for (let y = 0; y < h; y++) {
    let acc = 0;
    const row = y * w;
    for (let x = -r; x <= r; x++) acc += src[row + Math.max(0, Math.min(w - 1, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / win;
      const add = src[row + Math.min(w - 1, x + r + 1)];
      const sub = src[row + Math.max(0, x - r)];
      acc += add - sub;
    }
  }
  // Vertical
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[Math.max(0, Math.min(h - 1, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / win;
      const add = tmp[Math.min(h - 1, y + r + 1) * w + x];
      const sub = tmp[Math.max(0, y - r) * w + x];
      acc += add - sub;
    }
  }
  return out;
}

function saliencyMap(
  eq: Uint8ClampedArray, w: number, h: number, mask: Uint8Array,
): Uint8ClampedArray {
  const grad = sobel(eq, w, h);
  // Local mean & variance via box blur.
  const f = new Float32Array(eq.length);
  const f2 = new Float32Array(eq.length);
  for (let i = 0; i < eq.length; i++) { f[i] = eq[i]; f2[i] = eq[i] * eq[i]; }
  const r = Math.max(4, Math.round(Math.min(w, h) * 0.025));
  const meanLocal = boxBlur(f, w, h, r);
  const meanSq = boxBlur(f2, w, h, r);

  const sal = new Float32Array(eq.length);
  let smax = 0;
  for (let i = 0; i < eq.length; i++) {
    if (!mask[i]) { sal[i] = 0; continue; }
    const variance = Math.max(0, meanSq[i] - meanLocal[i] * meanLocal[i]);
    const std = Math.sqrt(variance);
    const dev = Math.abs(eq[i] - meanLocal[i]); // anomaly
    // Combine: edges + anomaly + texture + brightness inside lung
    const v = grad[i] * 0.4 + dev * 1.2 + std * 0.5 + Math.max(0, eq[i] - 90) * 0.3;
    sal[i] = v;
    if (v > smax) smax = v;
  }
  // Smooth a bit so lesions form contiguous blobs.
  const smoothed = boxBlur(sal, w, h, Math.max(2, (r >> 1)));
  let max = 0;
  for (const v of smoothed) if (v > max) max = v;
  const out = new Uint8ClampedArray(eq.length);
  if (max > 0) {
    for (let i = 0; i < smoothed.length; i++) {
      out[i] = mask[i] ? Math.min(255, (smoothed[i] / max) * 255) : 0;
    }
  }
  return out;
}

// -------------------- candidate extraction --------------------

interface Component {
  pixels: Int32Array;
  count: number;
  minX: number; minY: number; maxX: number; maxY: number;
  sumX: number; sumY: number;
}
function connectedComponents(bin: Uint8Array, w: number, h: number): Component[] {
  const labels = new Int32Array(bin.length);
  const comps: Component[] = [];
  const stack: number[] = [];
  let next = 1;
  for (let i = 0; i < bin.length; i++) {
    if (bin[i] !== 1 || labels[i]) continue;
    const buf: number[] = [];
    stack.push(i); labels[i] = next;
    let minX = w, minY = h, maxX = 0, maxY = 0, sumX = 0, sumY = 0;
    while (stack.length) {
      const p = stack.pop()!;
      buf.push(p);
      const px = p % w, py = (p / w) | 0;
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      sumX += px; sumY += py;
      const nbrs = [p - 1, p + 1, p - w, p + w];
      for (const n of nbrs) {
        if (n < 0 || n >= bin.length) continue;
        if (bin[n] === 1 && !labels[n]) { labels[n] = next; stack.push(n); }
      }
    }
    comps.push({
      pixels: Int32Array.from(buf),
      count: buf.length,
      minX, minY, maxX, maxY, sumX, sumY,
    });
    next++;
  }
  return comps;
}

function perimeter(bin: Uint8Array, w: number, h: number, c: Component): number {
  let p = 0;
  for (let k = 0; k < c.count; k++) {
    const i = c.pixels[k];
    const x = i % w, y = (i / w) | 0;
    const nbrs = [
      y > 0 ? i - w : -1,
      y < h - 1 ? i + w : -1,
      x > 0 ? i - 1 : -1,
      x < w - 1 ? i + 1 : -1,
    ];
    for (const n of nbrs) {
      if (n === -1 || bin[n] !== 1) { p++; break; }
    }
  }
  return p;
}

function convexHullArea(pts: Array<[number, number]>): number {
  if (pts.length < 3) return pts.length;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: number[][] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = lower.concat(upper.slice(0, -1));
  let a = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % hull.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

// Jet-style colormap.
function heat(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 1)));
  return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
}

// -------------------- main pipeline --------------------

export function runPipelineRGBA(
  rgba: Uint8ClampedArray, w: number, h: number,
): PipelineOutput {
  const gray = rgbaToGray(rgba);
  const eq = clahe(gray, w, h, 8, 8, 0.012);
  const mask = lungMask(eq, w, h);
  const sal = saliencyMap(eq, w, h, mask);

  // Threshold saliency (Otsu within mask).
  const tsal = otsu(sal, mask);
  const minThr = Math.max(tsal, 110);
  const bin = new Uint8Array(sal.length);
  for (let i = 0; i < sal.length; i++) bin[i] = (mask[i] && sal[i] >= minThr) ? 1 : 0;

  // Morph clean.
  let cleaned = morph(bin, w, h, "erode", 1);
  cleaned = morph(cleaned, w, h, "dilate", 2);

  const comps = connectedComponents(cleaned, w, h);

  // Lung area for ratio.
  let lungArea = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) lungArea++;
  lungArea = Math.max(lungArea, w * h * 0.05);

  // Filter & score candidates.
  const minA = Math.max(15, w * h * 0.0003);
  const maxA = w * h * 0.15;

  let best: Component | undefined;
  let bestScore = -Infinity;
  let bestFeatures: RegionFeatures | undefined;

  for (const c of comps) {
    if (c.count < minA || c.count > maxA) continue;
    const bw = c.maxX - c.minX + 1, bh = c.maxY - c.minY + 1;
    const ar = Math.max(bw, bh) / Math.max(1, Math.min(bw, bh));
    if (ar > 5) continue; // skip vessel-like

    const peri = perimeter(cleaned, w, h, c);
    const circ = (4 * Math.PI * c.count) / Math.max(1, peri * peri);
    const pts: Array<[number, number]> = new Array(c.count);
    for (let k = 0; k < c.count; k++) {
      const i = c.pixels[k];
      pts[k] = [i % w, (i / w) | 0];
    }
    const hullA = convexHullArea(pts);
    const solidity = c.count / Math.max(1, hullA);

    let sum = 0, sumSq = 0, salSum = 0;
    for (let k = 0; k < c.count; k++) {
      const i = c.pixels[k];
      sum += eq[i]; sumSq += eq[i] * eq[i]; salSum += sal[i];
    }
    const mean = sum / c.count;
    const variance = sumSq / c.count - mean * mean;
    const std = Math.sqrt(Math.max(0, variance));
    const salMean = (salSum / c.count) / 255;

    const ringPad = Math.max(4, Math.min(bw, bh) >> 1);
    let ringSum = 0, ringN = 0;
    for (let y = Math.max(0, c.minY - ringPad); y <= Math.min(h - 1, c.maxY + ringPad); y++) {
      for (let x = Math.max(0, c.minX - ringPad); x <= Math.min(w - 1, c.maxX + ringPad); x++) {
        const inside = x >= c.minX && x <= c.maxX && y >= c.minY && y <= c.maxY;
        if (!inside && mask[y * w + x]) { ringSum += eq[y * w + x]; ringN++; }
      }
    }
    const ringMean = ringN ? ringSum / ringN : mean;
    const contrast = Math.min(1, Math.abs(mean - ringMean) / 128);

    const feats: RegionFeatures = {
      area: c.count,
      areaRatio: c.count / lungArea,
      meanIntensity: mean,
      stdIntensity: std,
      circularity: Math.min(1, circ),
      solidity: Math.min(1, solidity),
      edgeIrregularity: Math.max(0, Math.min(1, 1 - circ)),
      contrast,
      saliencyMean: salMean,
      bbox: { x: c.minX, y: c.minY, w: bw, h: bh },
      centroid: { x: c.sumX / c.count, y: c.sumY / c.count },
      perimeter: peri,
    };

    const score =
      Math.log(c.count) * 0.5 +
      salMean * 2.5 +
      contrast * 1.6 +
      (mean / 255) * 0.4 +
      circ * 0.3;

    if (score > bestScore) { bestScore = score; best = c; bestFeatures = feats; }
  }

  // Build pre-rendered RGBA outputs.
  const overlayRGBA = new Uint8ClampedArray(rgba.length);
  overlayRGBA.set(rgba);
  // Heat overlay on the equalized image.
  const heatRGBA = new Uint8ClampedArray(rgba.length);
  for (let i = 0, j = 0; j < eq.length; i += 4, j++) {
    const base = eq[j];
    const t = sal[j] / 255;
    if (t < 0.05) {
      heatRGBA[i] = heatRGBA[i + 1] = heatRGBA[i + 2] = base;
    } else {
      const [r, g, b] = heat(t);
      heatRGBA[i] = (base * (1 - t) + r * t) | 0;
      heatRGBA[i + 1] = (base * (1 - t) + g * t) | 0;
      heatRGBA[i + 2] = (base * (1 - t) + b * t) | 0;
    }
    heatRGBA[i + 3] = 255;
  }

  if (best && bestFeatures) {
    // Overlay the best ROI bbox + outline on the original.
    const set = new Set<number>();
    for (let k = 0; k < best.count; k++) set.add(best.pixels[k]);
    // Edge pixels in red.
    for (let k = 0; k < best.count; k++) {
      const i = best.pixels[k];
      const nbrs = [i - 1, i + 1, i - w, i + w];
      let edge = false;
      for (const n of nbrs) if (n < 0 || n >= eq.length || !set.has(n)) { edge = true; break; }
      if (edge) {
        const idx = i * 4;
        overlayRGBA[idx] = 255; overlayRGBA[idx + 1] = 64; overlayRGBA[idx + 2] = 64;
      }
    }
    // BBox cyan.
    const drawPx = (x: number, y: number) => {
      if (x < 0 || y < 0 || x >= w || y >= h) return;
      const idx = (y * w + x) * 4;
      overlayRGBA[idx] = 0; overlayRGBA[idx + 1] = 230; overlayRGBA[idx + 2] = 230;
    };
    for (let x = best.minX; x <= best.maxX; x++) { drawPx(x, best.minY); drawPx(x, best.maxY); }
    for (let y = best.minY; y <= best.maxY; y++) { drawPx(best.minX, y); drawPx(best.maxX, y); }
  }

  // Growth heuristic.
  let growthScore = 0;
  if (bestFeatures) {
    const f = bestFeatures;
    growthScore =
      Math.min(1, f.areaRatio * 6) * 0.25 +
      f.edgeIrregularity * 0.3 +
      f.contrast * 0.2 +
      f.saliencyMean * 0.15 +
      Math.min(1, f.stdIntensity / 60) * 0.1;
    growthScore = Math.min(1, Math.max(0, growthScore));
  }
  const growthLabel =
    !bestFeatures ? "No region — growth N/A"
    : growthScore < 0.25 ? "Stable / low growth signal"
    : growthScore < 0.5 ? "Mild growth indicators"
    : growthScore < 0.75 ? "Moderate growth indicators"
    : "Strong growth indicators";

  // Padded bbox for cropping on main thread.
  let regionBBoxPad: PipelineOutput["regionBBoxPad"] = null;
  if (bestFeatures) {
    const pad = Math.max(8, Math.round(Math.max(bestFeatures.bbox.w, bestFeatures.bbox.h) * 0.25));
    const x = Math.max(0, bestFeatures.bbox.x - pad);
    const y = Math.max(0, bestFeatures.bbox.y - pad);
    const cw = Math.min(w - x, bestFeatures.bbox.w + pad * 2);
    const ch = Math.min(h - y, bestFeatures.bbox.h + pad * 2);
    regionBBoxPad = { x, y, w: cw, h: ch };
  }

  // Build single-channel grayscale RGBA stand-in arrays for visualization is done on main thread.
  return {
    width: w, height: h,
    grayscale: gray,
    equalized: eq,
    lungMask: new Uint8ClampedArray(mask.map((v) => v ? 255 : 0)),
    saliency: sal,
    overlayRGBA,
    heatRGBA,
    region: bestFeatures,
    regionBBoxPad,
    growthScore,
    growthLabel,
  };
}
