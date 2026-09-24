import { useCallback, useEffect, useRef, useState } from "react";

import { Field, fieldClass } from "@/components/Field";
import { RefreshButton } from "@/components/RefreshButton";
import { Button } from "@/components/ui/button";
import { WorkflowStep } from "@/components/WorkflowSteps";
import {
  canSetupGvhmrInDesktop,
  formatFileSize,
  getGvhmrRuntimeStatus,
  invalidateGvhmrRuntimeStatus,
  isSupportedVideoName,
  parseOptionalFocalLength,
  setupGvhmrInDesktop,
  startVideoToMotion,
  summarizeMotionResult,
  visibleGvhmrFailure,
  waitForVideoToMotion,
  type GvhmrRuntimeStatus,
  type MotionResultSummary,
} from "@/features/video-to-motion/api";
import { SmplxModelLinks } from "@/features/video-to-motion/SmplxModelLinks";
import type { MotionLibraryEntry } from "@/features/motion/api";
import type { UploadFile } from "@/lib/api";
import { useLocaleText } from "@/LocaleProvider";

import { FileImport, StatusMessage } from "./BatchParts";
import { publishedMotionEntry, uploadFileKey } from "./model";

type VideoStatus = "queued" | "uploading" | "running" | "done" | "error";

interface VideoItem {
  readonly id: string;
  readonly file: UploadFile;
  readonly status: VideoStatus;
  readonly progress: number;
  readonly message: string;
  readonly result?: MotionResultSummary;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function statusLabel(
  status: VideoStatus,
  text: (english: string, chinese: string) => string,
): string {
  if (status === "queued") return text("Queued", "排队中");
  if (status === "uploading") return text("Uploading", "上传中");
  if (status === "running") return text("Generating", "生成中");
  if (status === "done") return text("Published", "已发布");
  return text("Failed", "失败");
}

export function VideoBatchView({
  onMotionPublished,
  runtimeRevision = 0,
}: {
  onMotionPublished(entry: MotionLibraryEntry): void;
  runtimeRevision?: number;
}) {
  const text = useLocaleText();
  const [videos, setVideos] = useState<readonly VideoItem[]>([]);
  const [runtime, setRuntime] = useState<GvhmrRuntimeStatus | null>(null);
  const [runtimeChecking, setRuntimeChecking] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [staticCamera, setStaticCamera] = useState(true);
  const [focalLength, setFocalLength] = useState("");
  const [busy, setBusy] = useState(false);
  const [setupBusy, setSetupBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const runtimeRequest = useRef<AbortController | null>(null);
  const runRequest = useRef<AbortController | null>(null);

  const refreshRuntime = useCallback((fresh = false) => {
    if (fresh) invalidateGvhmrRuntimeStatus();
    runtimeRequest.current?.abort();
    const request = new AbortController();
    runtimeRequest.current = request;
    setRuntimeChecking(true);
    setRuntimeError(null);
    setConfirmed(false);
    void getGvhmrRuntimeStatus(request.signal)
      .then((status) => {
        if (!request.signal.aborted) setRuntime(status);
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) {
          setRuntime(null);
          setRuntimeError(errorMessage(reason));
        }
      })
      .finally(() => {
        if (!request.signal.aborted) setRuntimeChecking(false);
      });
  }, []);

  useEffect(() => {
    refreshRuntime();
    return () => {
      runtimeRequest.current?.abort();
      runRequest.current?.abort();
    };
  }, [refreshRuntime, runtimeRevision]);

  function addVideos(files: readonly UploadFile[]): void {
    let rejected = 0;
    const known = new Set(videos.map((item) => item.id));
    const additions: VideoItem[] = [];
    for (const file of files) {
      if (!isSupportedVideoName(file.name)) {
        rejected += 1;
        continue;
      }
      const id = uploadFileKey(file);
      if (known.has(id)) continue;
      known.add(id);
      additions.push({ id, file, status: "queued", progress: 0, message: "" });
    }
    setVideos((current) => {
      const currentKeys = new Set(current.map((item) => item.id));
      return [...current, ...additions.filter((item) => !currentKeys.has(item.id))];
    });
    setNotice(
      rejected
        ? text(
            `${rejected} unsupported file${rejected === 1 ? " was" : "s were"} skipped.`,
            `已跳过 ${rejected} 个不支持的文件。`,
          )
        : "",
    );
  }

  function patchVideo(id: string, patch: Partial<VideoItem>): void {
    setVideos((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }

  async function configureRuntime(): Promise<void> {
    setSetupBusy(true);
    setRuntimeError(null);
    try {
      const result = await setupGvhmrInDesktop();
      if (result.action === "configured") refreshRuntime(true);
    } catch (reason) {
      setRuntimeError(errorMessage(reason));
    } finally {
      setSetupBusy(false);
    }
  }

  let focalError: string | null = null;
  try {
    parseOptionalFocalLength(focalLength);
  } catch (reason) {
    focalError = text(errorMessage(reason), "焦距必须是正整数。");
  }
  const unfinished = videos.filter((item) => item.status !== "done");
  const completedCount = videos.length - unfinished.length;
  const failedCount = videos.filter((item) => item.status === "error").length;
  const aggregateProgress = videos.length
    ? videos.reduce((sum, item) => sum + item.progress, 0) / videos.length
    : 0;
  const disabledReason = busy
    ? text("The video queue is running.", "视频队列正在运行。")
    : !videos.length
      ? text("Add at least one video.", "请至少添加一个视频。")
      : !unfinished.length
        ? text("Every video is already complete.", "所有视频均已处理完成。")
        : runtimeChecking
          ? text("Checking the GVHMR runtime…", "正在检查 GVHMR 运行环境…")
          : runtime?.ready !== true
            ? text("The GVHMR runtime is unavailable.", "GVHMR 运行环境不可用。")
            : !confirmed
              ? text("Confirm the runtime before starting.", "请先确认运行环境。")
              : focalError;

  async function runQueue(): Promise<void> {
    if (disabledReason) return;
    const pending = videos.filter((item) => item.status !== "done");
    const parsedFocalLength = parseOptionalFocalLength(focalLength);
    runRequest.current?.abort();
    const request = new AbortController();
    runRequest.current = request;
    setBusy(true);
    setNotice(
      text(
        `Processing ${pending.length} video${pending.length === 1 ? "" : "s"} sequentially…`,
        `正在依次处理 ${pending.length} 个视频…`,
      ),
    );
    let completed = completedCount;
    let failed = 0;
    try {
      // One official GVHMR process at a time keeps GPU memory and attribution deterministic.
      for (let index = 0; index < pending.length; index += 1) {
        const item = pending[index];
        patchVideo(item.id, {
          status: "uploading",
          progress: 0.04,
          message: text("Uploading video…", "正在上传视频…"),
          result: undefined,
        });
        try {
          const jobId = await startVideoToMotion(
            { video: item.file, staticCamera, focalLength: parsedFocalLength },
            request.signal,
          );
          if (request.signal.aborted) return;
          patchVideo(item.id, {
            status: "running",
            progress: 0.08,
            message: text("Starting GVHMR…", "正在启动 GVHMR…"),
          });
          const motion = await waitForVideoToMotion(jobId, {
            signal: request.signal,
            onUpdate: (job) => {
              patchVideo(item.id, {
                status: "running",
                progress: 0.08 + job.progress * 0.92,
                message: job.message || text("Generating motion…", "正在生成动作…"),
              });
            },
          });
          if (request.signal.aborted) return;
          const result = summarizeMotionResult(motion, item.file.name);
          const libraryEntry = publishedMotionEntry(motion.library_entry);
          if (libraryEntry) onMotionPublished(libraryEntry);
          patchVideo(item.id, {
            status: "done",
            progress: 1,
            message: result.linkedFolder
              ? text(`Published to ${result.linkedFolder}`, `已发布到 ${result.linkedFolder}`)
              : text("Published to Motion Library", "已发布到动作资源库"),
            result,
          });
          completed += 1;
        } catch (reason) {
          if (request.signal.aborted) return;
          failed += 1;
          patchVideo(item.id, {
            status: "error",
            progress: 0,
            message: visibleGvhmrFailure(reason, text),
          });
        }
        setNotice(
          text(
            `Processed ${index + 1} of ${pending.length}.`,
            `已处理 ${index + 1}/${pending.length}。`,
          ),
        );
      }
      setNotice(
        text(
          `${completed} completed · ${failed} failed. Generated motions are in Motion Library.`,
          `${completed} 个完成 · ${failed} 个失败。生成的动作已存入动作资源库。`,
        ),
      );
    } finally {
      if (runRequest.current === request) {
        runRequest.current = null;
        setBusy(false);
      }
    }
  }

  const runtimeLabel = runtimeChecking
    ? text("Checking", "检查中")
    : runtime?.ready
      ? `${text("Ready", "就绪")} · ${runtime.runtime === "docker" ? "Docker" : text("Local", "本地")}`
      : text("Unavailable", "不可用");

  return (
    <div className="flex flex-col">
      <WorkflowStep
        title={text("1. Videos", "1. 视频")}
        status={text(`${videos.length} videos`, `${videos.length} 个视频`)}
        defaultOpen
      >
        <div className="grid gap-2.5">
          <FileImport
            title={text("Drop videos or a folder", "拖放视频或文件夹")}
            hint={text(
              "MP4, MOV, MKV, AVI, WebM or M4V",
              "MP4、MOV、MKV、AVI、WebM 或 M4V",
            )}
            icon="/icons/sidebar/video-to-motion.svg"
            accept="video/mp4,video/quicktime,video/x-matroska,video/x-msvideo,video/webm,.m4v"
            busy={busy}
            onFiles={addVideos}
          />
          <div className="max-h-56 overflow-y-auto rounded-md border border-border-subtle bg-surface">
            {videos.length ? videos.map((item) => (
              <div key={item.id} className="grid grid-cols-[minmax(0,1fr)_auto] gap-2 border-b border-border-subtle px-2.5 py-2 last:border-b-0">
                <div className="min-w-0">
                  <p className="truncate text-xs font-semibold text-foreground" title={item.file.name}>{item.file.name}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {formatFileSize(item.file.size)} · {statusLabel(item.status, text)}
                  </p>
                  {item.message && <p className={`mt-1 break-words text-[10px] ${item.status === "error" ? "text-danger" : "text-muted-foreground"}`}>{item.message}</p>}
                  {(item.status === "uploading" || item.status === "running") && (
                    <progress className="mt-1 h-1 w-full accent-primary" value={item.progress} max="1" />
                  )}
                </div>
                <button
                  type="button"
                  className="size-7 rounded-md text-lg leading-none text-muted-foreground hover:bg-danger-muted hover:text-danger disabled:cursor-not-allowed disabled:opacity-50"
                  aria-label={text(`Remove ${item.file.name}`, `移除 ${item.file.name}`)}
                  title={text("Remove", "移除")}
                  disabled={busy}
                  onClick={() => setVideos((current) => current.filter((candidate) => candidate.id !== item.id))}
                >
                  ×
                </button>
              </div>
            )) : (
              <p className="px-3 py-5 text-center text-xs text-muted-foreground">
                {text("No videos yet.", "尚未添加视频。")}
              </p>
            )}
          </div>
          <div className="flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
            <span>
              {completedCount} {text("ready", "就绪")} · {failedCount} {text("failed", "失败")}
            </span>
            <Button
              size="sm"
              variant="dangerOutline"
              disabled={busy || !videos.length}
              onClick={() => {
                setVideos([]);
                setNotice("");
              }}
            >
              {text("Clear all", "全部清除")}
            </Button>
          </div>
        </div>
      </WorkflowStep>

      <WorkflowStep title={text("2. Environment", "2. 运行环境")} status={runtimeLabel} defaultOpen>
        <div className="grid gap-2.5">
          <Field label={text("Runtime", "运行环境")}>
            <select className={fieldClass} value="official" disabled>
              <option value="official">{text("GVHMR Official", "GVHMR 官方版")}</option>
            </select>
          </Field>
          <div className="grid grid-cols-[minmax(0,1fr)_30px] gap-2">
            <Button size="sm" disabled={runtimeChecking || busy || runtime?.ready !== true || confirmed} onClick={() => setConfirmed(true)}>
              {confirmed ? text("Confirmed", "已确认") : text("Confirm", "确认")}
            </Button>
            <RefreshButton
              label={text("Refresh GVHMR status", "刷新 GVHMR 状态")}
              busy={runtimeChecking}
              variant="ghost"
              disabled={busy}
              onClick={() => refreshRuntime(true)}
            />
          </div>
          {canSetupGvhmrInDesktop() && runtime?.ready !== true && (
            <Button size="sm" disabled={setupBusy || busy} onClick={() => void configureRuntime()}>
              {setupBusy ? text("Setting up…", "配置中…") : text("Set up GVHMR", "配置 GVHMR")}
            </Button>
          )}
          <StatusMessage error={Boolean(runtimeError || (runtime && !runtime.ready))}>
            {runtimeError ||
              (runtime?.ready
                ? text("Official runtime and weights are ready.", "官方运行环境和权重已就绪。")
                : runtime?.missing?.[0])}
          </StatusMessage>
          {runtime && runtime.missing.length > 1 && (
            <details className="text-[11px] text-muted-foreground">
              <summary className="cursor-pointer">
                {text(
                  `${runtime.missing.length - 1} more checks`,
                  `另有 ${runtime.missing.length - 1} 项检查`,
                )}
              </summary>
              <ul className="mt-1 grid list-disc gap-1 pl-4">
                {runtime.missing.slice(1).map((item) => <li key={item}>{item}</li>)}
              </ul>
            </details>
          )}
          <SmplxModelLinks runtime={runtime} />
        </div>
      </WorkflowStep>

      <WorkflowStep
        title={text("3. Generate motions", "3. 生成动作")}
        status={
          busy
            ? `${Math.round(aggregateProgress * 100)}%`
            : unfinished.length
              ? text(`${unfinished.length} pending`, `${unfinished.length} 个待处理`)
              : videos.length
                ? text("Done", "完成")
                : text("Waiting", "等待中")
        }
        defaultOpen
      >
        <div className="grid gap-2.5">
          <label className="flex min-h-8 items-center justify-between gap-3 text-xs font-medium text-foreground">
            {text("Static camera", "静态相机")}
            <input type="checkbox" className="size-4 accent-primary" checked={staticCamera} disabled={busy} onChange={(event) => setStaticCamera(event.currentTarget.checked)} />
          </label>
          <Field label={text("Focal length", "焦距")}>
            <input className={fieldClass} type="number" min="1" step="1" placeholder={text("Auto", "自动")} value={focalLength} disabled={busy} onChange={(event) => setFocalLength(event.currentTarget.value)} />
          </Field>
          <Button variant="primary" size="sm" disabled={Boolean(disabledReason)} onClick={() => void runQueue()}>
            {busy
              ? text("Generating…", "生成中…")
              : failedCount || completedCount
                ? text("Retry unfinished", "重试未完成项")
                : text("Start V2M batch", "启动 V2M 批处理")}
          </Button>
          <progress className="h-1.5 w-full accent-primary" value={aggregateProgress} max="1" />
          {disabledReason && <p className="text-[11px] text-muted-foreground">{disabledReason}</p>}
          <StatusMessage error={Boolean(focalError)}>{focalError}</StatusMessage>
          {!focalError && <StatusMessage>{notice}</StatusMessage>}
        </div>
      </WorkflowStep>
    </div>
  );
}
