import { useRef, type ChangeEvent, type ReactNode } from "react";

import { Field, fieldClass } from "@/components/Field";
import { ImportDropzone } from "@/components/ImportDropzone";
import { SearchField } from "@/components/SearchField";
import { Button } from "@/components/ui/button";
import type { MotionLibraryEntry } from "@/features/motion/api";
import type { RobotSummary } from "@/features/robot/api";
import type { JobSnapshot, UploadFile } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useLocaleText } from "@/LocaleProvider";

import {
  batchDownloadUrl,
  type BatchBackend,
  type BatchFailure,
  type BatchFormat,
  type BatchResult,
} from "./api";
import { entryKey, entryReference, entryTitle } from "./model";

export function StatusMessage({
  children,
  error = false,
}: {
  children?: ReactNode;
  error?: boolean;
}) {
  if (!children) return null;
  return (
    <p
      className={cn(
        "rounded-md border px-2.5 py-2 text-[11px] leading-relaxed [overflow-wrap:anywhere]",
        error
          ? "border-danger-border bg-danger-muted text-danger"
          : "border-border-subtle bg-background text-muted-foreground",
      )}
      role={error ? "alert" : "status"}
    >
      {children}
    </p>
  );
}

export function RobotSelect({
  label,
  robots,
  value,
  loadedName,
  busy,
  onChange,
  onLoad,
}: {
  label: string;
  robots: readonly RobotSummary[];
  value: string;
  loadedName?: string | null;
  busy: boolean;
  onChange(value: string): void;
  onLoad(): void;
}) {
  const text = useLocaleText();
  const loaded = Boolean(value && value === loadedName);
  return (
    <div className="grid gap-2">
      <select
        className={fieldClass}
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
        disabled={busy}
      >
        <option value="">{text("Select a robot…", "选择机器人…")}</option>
        {robots.map((robot) => (
          <option key={robot.name} value={robot.name} disabled={!robot.has_urdf}>
            {robot.display_name || robot.name} ({robot.num_dof} {text("DoF", "自由度")})
          </option>
        ))}
      </select>
      <Button size="sm" onClick={onLoad} disabled={busy || !value || loaded}>
        {loaded ? text("Robot loaded", "机器人已加载") : text("Load robot", "加载机器人")}
      </Button>
    </div>
  );
}

export function FileImport({
  title,
  hint,
  icon,
  accept,
  busy,
  onFiles,
}: {
  title: string;
  hint: string;
  icon: string;
  accept?: string;
  busy: boolean;
  onFiles(files: readonly UploadFile[]): void | Promise<void>;
}) {
  const text = useLocaleText();
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const receive = (event: ChangeEvent<HTMLInputElement>) => {
    const files = [...(event.currentTarget.files ?? [])] as UploadFile[];
    event.currentTarget.value = "";
    if (files.length) void onFiles(files);
  };
  return (
    <ImportDropzone
      label={title}
      icon={icon}
      title={title}
      hint={hint}
      disabled={busy}
      onFiles={onFiles}
    >
      <input
        ref={fileInput}
        className="hidden"
        type="file"
        multiple
        accept={accept}
        disabled={busy}
        onChange={receive}
      />
      <input
        ref={folderInput}
        className="hidden"
        type="file"
        multiple
        disabled={busy}
        onChange={receive}
        {...({ webkitdirectory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
      />
      <Button size="sm" disabled={busy} onClick={() => fileInput.current?.click()}>
        {text("Files", "文件")}
      </Button>
      <Button size="sm" disabled={busy} onClick={() => folderInput.current?.click()}>
        {text("Folder", "文件夹")}
      </Button>
    </ImportDropzone>
  );
}

export function EntryList({
  entries,
  busy,
  kind,
  onRemove,
  onClear,
}: {
  entries: readonly MotionLibraryEntry[];
  busy: boolean;
  kind: "human" | "robot";
  onRemove(key: string): void;
  onClear(): void;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2">
      <div className="max-h-48 overflow-y-auto rounded-md border border-border-subtle bg-surface">
        {entries.length ? (
          entries.map((entry) => (
            <div
              key={entryKey(entry)}
              className="grid min-h-11 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 border-b border-border-subtle px-2.5 py-2 last:border-b-0"
            >
              <div className="min-w-0">
                <p className="truncate text-xs font-semibold text-foreground" title={entryTitle(entry)}>
                  {entryTitle(entry)}
                </p>
                <p className="truncate text-[10px] text-muted-foreground">
                  {kind === "human"
                    ? `${entry.motion_category ?? text("motion", "动作")} · ${entryReference(entry).toUpperCase()}`
                    : entry.upload_profile ?? text("auto", "自动")}
                </p>
              </div>
              <button
                type="button"
                className="size-7 rounded-md text-lg leading-none text-muted-foreground hover:bg-danger-muted hover:text-danger disabled:cursor-not-allowed disabled:opacity-50"
                aria-label={text(`Remove ${entryTitle(entry)}`, `移除 ${entryTitle(entry)}`)}
                title={text("Remove", "移除")}
                disabled={busy}
                onClick={() => onRemove(entryKey(entry))}
              >
                ×
              </button>
            </div>
          ))
        ) : (
          <p className="px-3 py-5 text-center text-xs text-muted-foreground">
            {text("No inputs yet.", "尚未添加输入。")}
          </p>
        )}
      </div>
      <div className="flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
        <span>
          {entries.length} {kind === "human" ? text("clips", "个动作") : text("trajectories", "条轨迹")}
        </span>
        <Button
          size="sm"
          variant="dangerOutline"
          disabled={busy || !entries.length}
          onClick={onClear}
        >
          {text("Clear all", "全部清除")}
        </Button>
      </div>
    </div>
  );
}

export function LibraryPicker({
  entries,
  selection,
  disabled,
  query,
  onQueryChange,
  onSelectionChange,
  onAdd,
}: {
  entries: readonly MotionLibraryEntry[];
  selection: ReadonlySet<string>;
  disabled: boolean;
  query: string;
  onQueryChange(value: string): void;
  onSelectionChange(value: ReadonlySet<string>): void;
  onAdd(): void;
}) {
  const text = useLocaleText();
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const visible = entries.filter((entry) => {
    const text = [entryTitle(entry), entry.folder_label, entry.dataset, entry.reference]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return tokens.every((token) => text.includes(token));
  });
  const toggle = (key: string) => {
    const next = new Set(selection);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    onSelectionChange(next);
  };
  return (
    <details className="rounded-md border border-border-subtle bg-background">
      <summary className="cursor-pointer list-none px-2.5 py-2 text-xs font-semibold text-foreground [&::-webkit-details-marker]:hidden">
        {text("Add from Motion Library", "从动作资源库添加")}
      </summary>
      <div className="grid gap-2 border-t border-border-subtle p-2.5">
        <SearchField
          label={text("Search Motion Library", "搜索动作资源库")}
          placeholder={text("Search motions…", "搜索动作…")}
          value={query}
          disabled={disabled}
          onChange={(event) => onQueryChange(event.currentTarget.value)}
        />
        <div className="max-h-40 overflow-y-auto rounded-md border border-border-subtle bg-surface">
          {visible.map((entry) => {
            const key = entryKey(entry);
            return (
              <label
                key={key}
                className="flex min-h-9 cursor-pointer items-center gap-2 border-b border-border-subtle px-2.5 py-1.5 last:border-b-0 hover:bg-accent"
              >
                <input
                  className="size-3.5 accent-primary"
                  type="checkbox"
                  checked={selection.has(key)}
                  disabled={disabled}
                  onChange={() => toggle(key)}
                />
                <span className="min-w-0 flex-1 truncate text-[11px] text-foreground">
                  {entryTitle(entry)}
                </span>
                <span className="shrink-0 text-[10px] text-muted-foreground">
                  {entryReference(entry).toUpperCase()}
                </span>
              </label>
            );
          })}
          {!visible.length && (
            <p className="px-3 py-4 text-center text-[11px] text-muted-foreground">
              {text("No matching motions.", "没有匹配的动作。")}
            </p>
          )}
        </div>
        <Button variant="primary" size="sm" onClick={onAdd} disabled={disabled || !selection.size}>
          {text(
            `Add ${selection.size || "selected"}`,
            selection.size ? `添加 ${selection.size} 项` : "添加所选项",
          )}
        </Button>
      </div>
    </details>
  );
}

export interface CommonBatchSettingsValue {
  backend: BatchBackend;
  format: BatchFormat;
  csvHeader: boolean;
  retargetFps: string;
  exportFps: string;
  start: string;
  end: string;
  output: string;
}

export function CommonBatchSettings({
  value,
  disabled,
  sourceFps,
  batchSize,
  onChange,
  onSourceFpsChange,
  onBatchSizeChange,
}: {
  value: CommonBatchSettingsValue;
  disabled: boolean;
  sourceFps?: string;
  batchSize?: string;
  onChange(patch: Partial<CommonBatchSettingsValue>): void;
  onSourceFpsChange?(value: string): void;
  onBatchSizeChange?(value: string): void;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2.5">
      <div className="grid grid-cols-2 gap-2">
        <Field label={text("Solver", "求解器")}>
          <select
            className={fieldClass}
            value={value.backend}
            disabled={disabled}
            onChange={(event) => onChange({ backend: event.currentTarget.value as BatchBackend })}
          >
            <option value="newton">Newton IK</option>
            <option value="interaction_mesh">Interaction-Mesh</option>
          </select>
        </Field>
        <Field label={text("Output format", "输出格式")}>
          <select
            className={fieldClass}
            value={value.format}
            disabled={disabled}
            onChange={(event) => onChange({ format: event.currentTarget.value as BatchFormat })}
          >
            <option value="pkl">PKL</option>
            <option value="csv">CSV</option>
          </select>
        </Field>
      </div>
      <details className="rounded-md border border-border-subtle bg-background">
        <summary className="cursor-pointer list-none px-2.5 py-2 text-xs font-semibold text-foreground [&::-webkit-details-marker]:hidden">
          {text("Advanced settings", "高级设置")}
        </summary>
        <div className="grid gap-2.5 border-t border-border-subtle p-2.5">
          {batchSize !== undefined && value.backend === "newton" && (
            <Field label={text("GPU batch size", "GPU 批大小")}>
              <input
                className={fieldClass}
                type="number"
                min="1"
                max="256"
                placeholder={text("Auto", "自动")}
                value={batchSize}
                disabled={disabled}
                onChange={(event) => onBatchSizeChange?.(event.currentTarget.value)}
              />
            </Field>
          )}
          <div className="grid grid-cols-2 gap-2">
            {sourceFps !== undefined && (
              <Field label={text("Source FPS", "源帧率")}>
                <input
                  className={fieldClass}
                  type="number"
                  min="1"
                  value={sourceFps}
                  disabled={disabled}
                  onChange={(event) => onSourceFpsChange?.(event.currentTarget.value)}
                />
              </Field>
            )}
            <Field label={text("Retarget FPS", "重定向帧率")}>
              <input
                className={fieldClass}
                type="number"
                min="1"
                placeholder={text("Source", "跟随源帧率")}
                value={value.retargetFps}
                disabled={disabled}
                onChange={(event) => onChange({ retargetFps: event.currentTarget.value })}
              />
            </Field>
            <Field label={text("Export FPS", "导出帧率")}>
              <input
                className={fieldClass}
                type="number"
                min="1"
                placeholder={text("Retarget", "跟随重定向帧率")}
                value={value.exportFps}
                disabled={disabled}
                onChange={(event) => onChange({ exportFps: event.currentTarget.value })}
              />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Field label={text("Start (s)", "开始时间（秒）")}>
              <input
                className={fieldClass}
                type="number"
                min="0"
                value={value.start}
                disabled={disabled}
                onChange={(event) => onChange({ start: event.currentTarget.value })}
              />
            </Field>
            <Field label={text("End (s)", "结束时间（秒）")}>
              <input
                className={fieldClass}
                type="number"
                min="0"
                placeholder={text("End", "结束")}
                value={value.end}
                disabled={disabled}
                onChange={(event) => onChange({ end: event.currentTarget.value })}
              />
            </Field>
          </div>
          {value.format === "csv" && (
            <label className="flex min-h-8 items-center justify-between gap-3 text-xs font-medium text-foreground">
              {text("Include CSV header", "包含 CSV 表头")}
              <input
                type="checkbox"
                className="size-4 accent-primary"
                checked={value.csvHeader}
                disabled={disabled}
                onChange={(event) => onChange({ csvHeader: event.currentTarget.checked })}
              />
            </label>
          )}
          <Field label={text("Result name", "结果名称")}>
            <input
              className={fieldClass}
              value={value.output}
              disabled={disabled}
              onChange={(event) => onChange({ output: event.currentTarget.value })}
            />
          </Field>
        </div>
      </details>
    </div>
  );
}

export function BatchProgress({ job }: { job: JobSnapshot<BatchResult> | null }) {
  const text = useLocaleText();
  if (!job) return null;
  const rows = [
    [text("Overall", "总体"), job.progress ?? 0],
    [text("Current", "当前"), job.clip_progress ?? 0],
  ] as const;
  return (
    <div className="grid gap-2" role="status" aria-live="polite">
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[52px_minmax(0,1fr)_32px] items-center gap-2 text-[10px] text-muted-foreground">
          <span>{label}</span>
          <progress className="h-1.5 w-full accent-primary" max="1" value={value} />
          <strong className="text-right text-foreground">{Math.round(value * 100)}%</strong>
        </div>
      ))}
      {job.message && <p className="text-[11px] text-muted-foreground">{job.message}</p>}
    </div>
  );
}

function FailureList({ failures }: { failures: readonly BatchFailure[] }) {
  const text = useLocaleText();
  if (!failures.length) return null;
  return (
    <details className="rounded-md border border-danger-border bg-danger-muted" open>
      <summary className="cursor-pointer list-none px-2.5 py-2 text-xs font-semibold text-danger [&::-webkit-details-marker]:hidden">
        {text("Failures", "失败项")} ({failures.length})
      </summary>
      <ul className="grid max-h-44 gap-2 overflow-y-auto border-t border-danger-border p-2.5">
        {failures.map((failure, index) => (
          <li key={`${failure.stem ?? "clip"}-${index}`} className="text-[11px] leading-relaxed text-danger">
            <strong>{failure.stem || text("Untitled clip", "未命名动作")}</strong>
            <span className="ml-1 rounded bg-surface/70 px-1 py-0.5 text-[9px] uppercase">
              {failure.stage || text("unknown", "未知")}
            </span>
            <p className="break-words">{failure.reason || text("Unknown error", "未知错误")}</p>
            {failure.log_rel && <code className="break-all">{failure.log_rel}</code>}
            {!failure.log_rel && failure.stash_error && (
              <p className="break-words">
                {text("Source copy failed:", "源文件复制失败：")} {failure.stash_error}
              </p>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function BatchResultPanel({
  jobId,
  result,
}: {
  jobId: string;
  result: BatchResult;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2.5">
      <div className="rounded-md border border-border-subtle bg-background p-2.5 text-[11px] text-muted-foreground">
        <p className="font-semibold text-foreground">
          {result.failures.length
            ? text("Completed with failures", "已完成，但存在失败项")
            : text("Batch complete", "批处理完成")}
        </p>
        <p className="mt-1">
          {result.written.length} {text("succeeded", "成功")} · {result.failures.length} {text("failed", "失败")}
          {result.solver_mode ? ` · ${result.solver_mode}` : ""}
        </p>
        {result.failure_log && (
          <p className="mt-1 break-all">
            {text("Failure data:", "失败数据：")} <code>{result.failure_log}</code>
          </p>
        )}
      </div>
      {result.download_name && (
        <a
          className="inline-flex min-h-[30px] min-w-0 items-center justify-center truncate rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:border-primary hover:bg-accent focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring"
          href={batchDownloadUrl(jobId)}
          download={result.download_name}
        >
          {text("Download ZIP", "下载 ZIP")}
        </a>
      )}
      <FailureList failures={result.failures} />
    </div>
  );
}
