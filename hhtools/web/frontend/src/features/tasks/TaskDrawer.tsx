import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useLocale, useLocaleText } from "@/LocaleProvider";
import { boundedProgress, displayFileName } from "@/lib/api";
import { cn } from "@/lib/utils";

import {
  canExportTaskResult,
  isWorkflowResultTask,
  listTasks,
  taskDownloadUrl,
  type TaskRecord,
  type TaskStatus,
} from "./api";
import { taskPollingDelay } from "./polling";

const KIND_LABELS: Readonly<Record<string, readonly [string, string]>> = {
  dataset_analyze: ["Dataset Analysis", "数据集分析"],
  dataset_robot_preview: ["Robot Preview", "机器人预览"],
  motion_load: ["Load Motion", "加载动作"],
  motion_link: ["Import Motion", "导入动作"],
  basket_upload: ["Import Batch Motions", "导入批量动作"],
  video_to_motion: ["Video to Motion", "视频转动作"],
  retarget: ["Human to Robot", "人体到机器人"],
  batch: ["Human to Robot Batch", "人体到机器人批处理"],
  r2r_source_upload: ["Load Robot Trajectory", "加载机器人轨迹"],
  r2r_retarget: ["Robot to Robot", "机器人到机器人"],
  r2r_basket_upload: ["Import Robot Batch", "导入机器人批处理"],
  r2r_batch: ["Robot to Robot Batch", "机器人到机器人批处理"],
};

const STATUS_LABELS: Readonly<Record<TaskStatus, readonly [string, string]>> = {
  pending: ["Pending", "等待中"],
  running: ["Running", "运行中"],
  done: ["Completed", "已完成"],
  error: ["Failed", "失败"],
};

const STATUS_STYLES: Readonly<
  Record<TaskStatus, { dot: string; label: string }>
> = {
  pending: { dot: "bg-muted-foreground", label: "text-muted-foreground" },
  running: { dot: "bg-primary", label: "text-primary" },
  done: { dot: "bg-success", label: "text-success" },
  error: { dot: "bg-danger", label: "text-danger" },
};

const TASK_ACTION_CLASS =
  "inline-flex items-center gap-0.5 self-center shrink-0 border-0 bg-transparent p-0 text-[11px] text-primary transition-colors hover:text-primary/80 focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:cursor-default disabled:text-muted-foreground disabled:opacity-40 disabled:hover:text-muted-foreground max-[560px]:col-start-2 max-[560px]:justify-self-start";

function ExportLabel({ text }: { readonly text: (en: string, zh: string) => string }) {
  return (
    <>
      <span>{text("Export", "导出")}</span>
      <span
        className="size-3 -rotate-90 bg-current [mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat]"
        aria-hidden="true"
      />
    </>
  );
}

const PARAMETER_LABELS: Readonly<Record<string, readonly [string, string]>> = {
  robot: ["Robot", "机器人"],
  target: ["Target", "目标机器人"],
  target_robot: ["Target", "目标机器人"],
  source_robot: ["Source", "源机器人"],
  profile: ["Profile", "配置"],
  reference: ["Skeleton", "参考骨架"],
  backend: ["Solver", "求解器"],
  embedding: ["Feature", "特征空间"],
  format: ["Format", "格式"],
  retarget_fps: ["Retarget FPS", "重定向 FPS"],
  export_fps: ["Export FPS", "导出 FPS"],
  source_fps: ["Source FPS", "源 FPS"],
  batch_size: ["Batch size", "批量数"],
  folder_label: ["Folder", "目录"],
  library_folder_label: ["Folder", "资源目录"],
  entry_count: ["Entries", "条目"],
  file_count: ["Files", "文件"],
};

function scalar(value: unknown): value is string | number | boolean {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function taskFacts(task: TaskRecord): readonly [string, unknown][] {
  // The allowlist deliberately excludes source/output paths from the UI.
  return Object.entries(task.parameters)
    .filter(([key, value]) => key in PARAMETER_LABELS && scalar(value))
    .slice(0, 5);
}

function resultSummary(
  task: TaskRecord,
  text: (en: string, zh: string) => string,
) {
  const parts: string[] = [];
  const success = task.result_summary.success_count;
  const failure = task.result_summary.failure_count;
  const frames = task.result_summary.num_frames;
  if (typeof success === "number") {
    parts.push(`${success} ${text("succeeded", "成功")}`);
  }
  if (typeof failure === "number" && failure > 0) {
    parts.push(`${failure} ${text("failed", "失败")}`);
  }
  if (typeof frames === "number") {
    parts.push(`${frames} ${text("frames", "帧")}`);
  }
  return parts.join(" · ");
}

function downloadName(task: TaskRecord): string | undefined {
  const value = task.result_summary.download_name;
  return typeof value === "string"
    ? displayFileName(value, "result")
    : undefined;
}

function formatDuration(
  seconds: number,
  text: (en: string, zh: string) => string,
) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) {
    return `${Math.max(1, Math.round(seconds))} ${text("sec", "秒")}`;
  }
  return `${Math.floor(seconds / 60)} ${text("min", "分")} ${Math.round(seconds % 60)} ${text("sec", "秒")}`;
}

/** Compact task history dock shared by the browser and Electron renderer. */
export function TaskDrawer() {
  const locale = useLocale();
  const text = useLocaleText();
  const [open, setOpen] = useState(false);
  const [tasks, setTasks] = useState<readonly TaskRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [consecutiveFailures, setConsecutiveFailures] = useState(0);
  const [pageVisible, setPageVisible] = useState(
    () => document.visibilityState !== "hidden",
  );
  const request = useRef<AbortController | null>(null);

  const refresh = useCallback(async (quiet = false) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    if (!quiet) setLoading(true);
    try {
      setTasks(await listTasks({ signal: controller.signal }));
      setError(null);
      setConsecutiveFailures(0);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : String(reason));
        setConsecutiveFailures((count) => Math.min(6, count + 1));
      }
    } finally {
      if (request.current === controller) {
        request.current = null;
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => {
      request.current?.abort();
    };
  }, [refresh]);

  const hasActiveTasks = tasks.some(
    (task) => task.status === "pending" || task.status === "running",
  );
  useEffect(() => {
    const delay = taskPollingDelay({
      visible: pageVisible,
      drawerOpen: open,
      hasActiveTasks,
      consecutiveFailures,
    });
    if (delay === null) return;
    const timer = window.setTimeout(() => void refresh(true), delay);
    return () => window.clearTimeout(timer);
  }, [consecutiveFailures, hasActiveTasks, open, pageVisible, refresh]);

  useEffect(() => {
    const onVisibilityChange = () => {
      const visible = document.visibilityState !== "hidden";
      setPageVisible(visible);
      if (visible && (open || hasActiveTasks)) void refresh(true);
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [hasActiveTasks, open, refresh]);

  useEffect(() => {
    const toggle = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "j") {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    window.addEventListener("keydown", toggle);
    return () => window.removeEventListener("keydown", toggle);
  }, []);

  const localized = (
    labels: readonly [string, string] | undefined,
    fallback: string,
  ) => (labels ? text(labels[0], labels[1]) : fallback);
  const timeFormat = useMemo(
    () => new Intl.DateTimeFormat(locale === "zh-CN" ? "zh-CN" : "en-US", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }),
    [locale],
  );

  return (
    <section
      className={cn(
        "col-start-2 row-start-3 z-40 min-w-0 max-[780px]:row-start-4",
        open
          ? "h-[min(42vh,360px)] max-[780px]:h-[min(30vh,240px)]"
          : "h-[34px]",
      )}
      aria-label={text("Tasks", "任务")}
    >
      {!open ? (
        <button
          type="button"
          className="flex h-full w-full items-center gap-2 border-x-0 border-t border-b-0 border-border-subtle bg-surface px-3 text-left text-xs font-semibold text-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring"
          aria-expanded="false"
          onClick={() => {
            setOpen(true);
            void refresh(true);
          }}
        >
          <span>{text("Tasks", "任务")}</span>
          <span
            className="ml-auto size-4 rotate-180 bg-current text-muted-foreground [mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat]"
            aria-hidden="true"
          />
        </button>
      ) : (
        <div className="flex h-full min-h-0 flex-col overflow-hidden border-x-0 border-t border-b-0 border-border-subtle bg-surface">
          <header className="flex min-h-11 items-center gap-3 border-b border-border-subtle px-3.5">
            <strong className="text-[13px] text-foreground">
              {text("Tasks", "任务")}
            </strong>
            <div className="ml-auto flex items-center gap-1">
              <button
                type="button"
                className="grid size-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                aria-label={text("Refresh tasks", "刷新任务")}
                title={text("Refresh tasks", "刷新任务")}
                onClick={() => void refresh()}
              >
                <span
                  className={cn(
                    "size-4 bg-current [mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat]",
                    loading && "animate-spin",
                  )}
                  aria-hidden="true"
                />
              </button>
              <button
                type="button"
                className="grid size-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                aria-label={text("Collapse tasks", "收起任务")}
                title={text("Collapse tasks", "收起任务")}
                aria-expanded="true"
                onClick={() => setOpen(false)}
              >
                <span
                  className="size-4 bg-current [mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat]"
                  aria-hidden="true"
                />
              </button>
            </div>
          </header>

          {error ? (
            <p
              className="m-auto px-4 text-center text-xs text-danger"
              role="alert"
            >
              {error}
            </p>
          ) : loading && tasks.length === 0 ? (
            <p className="m-auto px-4 text-center text-xs text-muted-foreground" role="status">
              {text("Loading tasks...", "正在读取任务...")}
            </p>
          ) : tasks.length === 0 ? (
            <p className="m-auto px-4 text-center text-xs text-muted-foreground">
              {text(
                "Completed workflows will appear here.",
                "运行工作流后，任务会显示在这里。",
              )}
            </p>
          ) : (
            <div className="min-h-0 overflow-y-auto overscroll-contain">
              {tasks.map((task) => {
                const facts = taskFacts(task);
                const summary = resultSummary(task, text);
                const statusStyle =
                  STATUS_STYLES[task.status] ?? STATUS_STYLES.pending;
                const createdAt = Number.isFinite(task.created_at)
                  ? timeFormat.format(new Date(task.created_at * 1_000))
                  : text("Unknown time", "时间未知");
                const workflowResult = isWorkflowResultTask(task);
                const canExport = canExportTaskResult(task);
                return (
                  <article
                    key={task.id}
                    className="grid grid-cols-[10px_minmax(0,1fr)_auto] items-start gap-2.5 border-b border-border-subtle px-3 py-2.5 last:border-b-0 max-[560px]:grid-cols-[10px_minmax(0,1fr)]"
                  >
                    <span
                      className={cn("mt-1.5 size-2 rounded-full", statusStyle.dot)}
                      aria-hidden="true"
                    />
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <strong className="text-xs text-foreground">
                          {localized(KIND_LABELS[task.kind], task.kind)}
                        </strong>
                        <span
                          className={cn(
                            "text-[10px] font-semibold",
                            statusStyle.label,
                          )}
                        >
                          {localized(STATUS_LABELS[task.status], task.status)}
                        </span>
                        <time className="text-[10px] tabular-nums text-muted-foreground">
                          {createdAt}
                        </time>
                        <span className="text-[10px] tabular-nums text-muted-foreground">
                          {formatDuration(task.duration_seconds, text)}
                        </span>
                      </div>
                      {(task.error || task.message) && (
                        <p
                          className={cn(
                            "mt-0.5 truncate text-[11px] text-muted-foreground",
                            task.error && "text-danger",
                          )}
                        >
                          {task.error || task.message}
                        </p>
                      )}
                      {task.status === "running" && (
                        <div
                          className="mt-1.5 h-[3px] overflow-hidden rounded-full bg-muted"
                          role="progressbar"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={Math.round(
                            boundedProgress(task.progress) * 100,
                          )}
                        >
                          <span
                            className="block h-full rounded-full bg-primary transition-[width]"
                            style={{
                              width: `${Math.round(boundedProgress(task.progress) * 100)}%`,
                            }}
                          />
                        </div>
                      )}
                      {(facts.length > 0 || summary) && (
                        <div className="mt-1 flex flex-wrap gap-x-2.5 gap-y-0.5 text-[10px] text-muted-foreground">
                          {facts.map(([key, value]) => (
                            <span key={key}>
                              {localized(PARAMETER_LABELS[key], key)}: {String(value)}
                            </span>
                          ))}
                          {summary && (
                            <span className="font-semibold text-foreground">
                              {summary}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    {workflowResult &&
                      (canExport ? (
                        <a
                          className={TASK_ACTION_CLASS}
                          href={taskDownloadUrl(task.id)}
                          download={downloadName(task)}
                        >
                          <ExportLabel text={text} />
                        </a>
                      ) : (
                        <button
                          type="button"
                          className={TASK_ACTION_CLASS}
                          disabled
                          title={text(
                            "Available when the workflow completes",
                            "工作流完成后可导出",
                          )}
                        >
                          <ExportLabel text={text} />
                        </button>
                      ))}
                  </article>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
