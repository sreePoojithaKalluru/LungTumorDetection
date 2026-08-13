interface Props {
  label: string;
  value: number; // 0..1
  display?: string;
  tone?: "primary" | "warning" | "destructive" | "success";
}

export function MetricBar({ label, value, display, tone = "primary" }: Props) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const toneVar = {
    primary: "var(--primary)",
    warning: "var(--warning)",
    destructive: "var(--destructive)",
    success: "var(--success)",
  }[tone];

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-sm tabular-nums" style={{ color: toneVar }}>
          {display ?? `${pct.toFixed(1)}%`}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${toneVar}, color-mix(in oklab, ${toneVar} 50%, transparent))`,
            boxShadow: `0 0 12px -2px ${toneVar}`,
          }}
        />
      </div>
    </div>
  );
}
