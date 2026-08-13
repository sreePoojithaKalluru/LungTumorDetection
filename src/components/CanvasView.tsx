import { useEffect, useRef } from "react";

interface Props {
  data: ImageData | null;
  label: string;
  hint?: string;
  className?: string;
}

export function CanvasView({ data, label, hint, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!ref.current || !data) return;
    ref.current.width = data.width;
    ref.current.height = data.height;
    ref.current.getContext("2d")!.putImageData(data, 0, 0);
  }, [data]);

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
          {label}
        </span>
        {hint && (
          <span className="text-[10px] font-mono text-primary/70">{hint}</span>
        )}
      </div>
      <div className="relative aspect-square bg-black/40 rounded-md overflow-hidden border border-border scanlines">
        {data ? (
          <canvas ref={ref} className="absolute inset-0 w-full h-full object-contain" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground/50 font-mono">
            — no signal —
          </div>
        )}
      </div>
    </div>
  );
}
