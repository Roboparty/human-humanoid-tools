import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { useLocaleText } from "@/LocaleProvider";

import {
  buildTrackingChart,
  topEffectors,
  type ResultDiagnosticsPayload,
} from "./model";
import {
  COMPARISON_PRESETS,
  isComparisonPreset,
  type ComparisonPreset,
} from "./comparison";

const PRESET_LABELS: Readonly<Record<ComparisonPreset, readonly [string, string]>> = {
  source: ["Source", "源"],
  target: ["Target", "目标"],
  result: ["Result", "结果"],
  overlay: ["Overlay", "叠加"],
};

function formatCm(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(1)} cm`;
}

function formatPercent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(0)}%`;
}

/** Shared H2R/R2R diagnostics rendered directly from the backend result DTO. */
export function ResultDiagnostics({
  diagnostics,
  preset,
  onPresetChange,
}: {
  readonly diagnostics?: ResultDiagnosticsPayload | null;
  readonly preset?: ComparisonPreset;
  readonly onPresetChange?: (preset: ComparisonPreset) => void;
}) {
  const text = useLocaleText();
  const resolved = diagnostics ?? {
    schema_version: 1,
    available: false,
    reason: text(
      "The server did not return tracking or contact diagnostics.",
      "服务器未返回跟踪或接触诊断。",
    ),
  };
  const tracking = resolved.tracking;
  const chart = buildTrackingChart(tracking?.series ?? []);
  const effectors = topEffectors(tracking?.effectors ?? []);
  const quality = !resolved.available || tracking == null
    ? {
        label: text("Unavailable", "不可用"),
        className: "bg-accent text-muted-foreground",
      }
    : tracking.p95_error_m <= 0.05
      ? { label: text("Stable", "稳定"), className: "bg-success/10 text-success" }
      : tracking.p95_error_m <= 0.1
        ? { label: text("Review", "需检查"), className: "bg-warning-muted text-warning" }
        : {
            label: text("Large deviation", "偏差较大"),
            className: "bg-danger-muted text-danger",
          };

  return (
    <section
      className="grid gap-3 rounded-md border border-border-subtle bg-background p-3"
      aria-label={text("Result diagnostics", "结果诊断")}
    >
      <header className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-xs font-semibold text-foreground">
            {text("Result diagnostics", "结果诊断")}
          </h3>
          <p className="mt-0.5 text-[11px] leading-[1.4] text-muted-foreground">
            {text("Preview-frame tracking and contact checks", "预览帧跟踪与接触检查")}
          </p>
        </div>
        <span
          className={`shrink-0 rounded px-2 py-1 text-[10px] font-semibold ${quality.className}`}
        >
          {quality.label}
        </span>
      </header>

      {preset && (
        <ToggleGroup
          type="single"
          value={preset}
          disabled={!onPresetChange}
          onValueChange={(value) => {
            if (isComparisonPreset(value)) onPresetChange?.(value);
          }}
          aria-label={text("Stage comparison", "舞台对比")}
          className="grid w-full grid-cols-4 overflow-hidden rounded-md border border-border-subtle bg-surface p-0.5"
        >
          {COMPARISON_PRESETS.map((option) => (
            <ToggleGroupItem
              key={option}
              value={option}
              className="min-h-7 min-w-0 rounded-sm px-1.5 text-[10px] font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground data-[state=on]:bg-primary data-[state=on]:font-semibold data-[state=on]:text-primary-foreground"
            >
              {text(...PRESET_LABELS[option])}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      )}

      {!resolved.available || !tracking ? (
        <p className="text-xs leading-[1.45] text-muted-foreground" role="status">
          {resolved.reason ||
            text(
              "This result does not contain enough mapped data for diagnostics.",
              "此结果没有足够的映射数据可用于诊断。",
            )}
        </p>
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border-subtle bg-border-subtle">
            <Metric label={text("Mean error", "平均误差")} value={formatCm(tracking.mean_error_m)} />
            <Metric label={text("P95 error", "P95 误差")} value={formatCm(tracking.p95_error_m)} />
            <Metric
              label={text("Contact agreement", "接触一致率")}
              value={resolved.contact?.available
                ? formatPercent(resolved.contact.agreement_ratio)
                : "—"}
            />
            <Metric
              label={text("Foot slide", "足部滑动")}
              value={resolved.contact?.available
                ? `${formatCm(resolved.contact.target_slide_mean_mps)}/s`
                : "—"}
            />
          </dl>

          {chart && (
            <div className="grid gap-1.5">
              <div className="flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
                <span>{text("Per-frame position error", "逐帧位置误差")}</span>
                <span>{text(`Peak ${formatCm(chart.peak)}`, `峰值 ${formatCm(chart.peak)}`)}</span>
              </div>
              <svg
                className="h-[72px] w-full overflow-hidden rounded-md bg-surface"
                viewBox={`0 0 ${chart.width} ${chart.height}`}
                preserveAspectRatio="none"
                role="img"
                aria-label={text(
                  "Per-frame mean and maximum position error",
                  "逐帧平均与最大位置误差",
                )}
              >
                <line
                  x1="0"
                  y1={chart.height - 1}
                  x2={chart.width}
                  y2={chart.height - 1}
                  stroke="currentColor"
                  className="text-border"
                />
                <polyline
                  points={chart.maxPoints}
                  fill="none"
                  stroke="#d16a61"
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                />
                <polyline
                  points={chart.meanPoints}
                  fill="none"
                  stroke="#0071e3"
                  strokeWidth="1.5"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
              <div className="flex gap-3 text-[10px] text-muted-foreground">
                <Legend color="bg-[#0071e3]" label={text("Mean", "平均")} />
                <Legend color="bg-[#d16a61]" label={text("Maximum", "最大")} />
              </div>
            </div>
          )}

          {resolved.contact && !resolved.contact.available && (
            <p className="text-[10px] leading-[1.45] text-muted-foreground">
              {text("Contact metrics unavailable:", "接触指标不可用：")} {resolved.contact.reason || text("foot mappings are missing", "缺少足部映射")}
              {text(".", "。")}
            </p>
          )}

          {effectors.length > 0 && (
            <div className="grid gap-1.5">
              <div className="flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
                <span>{text("Largest mapped-point deviations", "映射点最大偏差")}</span>
                <span>
                  {text(
                    `${resolved.mapped_effectors ?? effectors.length}/${resolved.requested_effectors ?? effectors.length} mapped`,
                    `已映射 ${resolved.mapped_effectors ?? effectors.length}/${resolved.requested_effectors ?? effectors.length} 个`,
                  )}
                </span>
              </div>
              <div className="divide-y divide-border-subtle border-y border-border-subtle">
                {effectors.map((effector) => (
                  <div
                    key={`${effector.canonical}:${effector.target_link}`}
                    className="flex items-center justify-between gap-3 py-1.5 text-[11px]"
                  >
                    <span
                      className="min-w-0 truncate"
                      title={effector.target_link}
                    >
                      {effector.canonical}
                    </span>
                    <strong className="shrink-0 font-semibold text-foreground">
                      {formatCm(effector.p95_error_m)}
                    </strong>
                  </div>
                ))}
              </div>
            </div>
          )}

          <p className="text-[10px] leading-[1.45] text-muted-foreground">
            {text(
              "Preview diagnostics flag tracking and contact anomalies; validate final motion in simulation before hardware use.",
              "预览诊断用于标记跟踪与接触异常；用于硬件前，请在仿真中验证最终动作。",
            )}
          </p>
        </>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
}: {
  readonly label: string;
  readonly value: string;
}) {
  return (
    <div className="grid min-w-0 gap-0.5 bg-surface p-2">
      <dt className="truncate text-[10px] text-muted-foreground">{label}</dt>
      <dd className="truncate text-xs font-semibold text-foreground">{value}</dd>
    </div>
  );
}

function Legend({
  color,
  label,
}: {
  readonly color: string;
  readonly label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1">
      <i className={`h-0.5 w-3 ${color}`} aria-hidden="true" />
      {label}
    </span>
  );
}
