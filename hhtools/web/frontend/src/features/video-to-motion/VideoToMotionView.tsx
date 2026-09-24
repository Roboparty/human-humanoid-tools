import { useCallback, useEffect, useRef, useState } from "react";

import { Field, fieldClass } from "@/components/Field";
import { ImportDropzone } from "@/components/ImportDropzone";
import { InspectorPage } from "@/components/Inspector";
import { RefreshButton } from "@/components/RefreshButton";
import { Button } from "@/components/ui/button";
import {
  WorkflowPipeline,
  WorkflowStep,
  type WorkflowStatusTone,
} from "@/components/WorkflowSteps";
import type { ApplicationImportRequest } from "@/importIntent";
import { useLocaleText } from "@/LocaleProvider";
import { cn } from "@/lib/utils";
import type { StageMotionPayload } from "@/stage/types";

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
  toStageMotionPayload,
  visibleGvhmrFailure,
  waitForVideoToMotion,
  type GvhmrRuntimeStatus,
  type MotionResultSummary,
  type VideoToMotionJob,
} from "./api";
import { SmplxModelLinks } from "./SmplxModelLinks";

type RuntimePhase = "checking" | "ready" | "unavailable" | "error";
type WorkflowPhase = "idle" | "uploading" | "running" | "done" | "error";
type WorkflowErrorOwner = "selection" | "generation";

interface SelectedVideo {
  readonly file: File;
  readonly previewUrl: string;
  readonly duration: number | null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatMetric(value: number | null, suffix = ""): string {
  if (value === null) return "--";
  const display = Number.isInteger(value) ? String(value) : value.toFixed(2);
  return `${display}${suffix}`;
}

export function VideoToMotionView({
  onMotionLoaded,
  onMotionLibraryChange,
  importRequest,
  runtimeRevision = 0,
}: {
  onMotionLoaded?: (motion: StageMotionPayload | null) => void;
  onMotionLibraryChange?: () => void;
  /** App-owned File-menu intent; this mounted view owns its input element. */
  importRequest?: ApplicationImportRequest | null;
  /** Settings increments this after configuring the shared GVHMR runtime. */
  runtimeRevision?: number;
}) {
  const text = useLocaleText();
  const pipeline = [
    text("Select Video", "选择视频"),
    text("Environment", "运行环境"),
    text("Generate", "生成动作"),
    text("Motion Result", "动作结果"),
  ];
  const [runtimePhase, setRuntimePhase] = useState<RuntimePhase>("checking");
  const [runtime, setRuntime] = useState<GvhmrRuntimeStatus | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [video, setVideo] = useState<SelectedVideo | null>(null);
  const [staticCamera, setStaticCamera] = useState(true);
  const [focalLength, setFocalLength] = useState("");
  const [workflowPhase, setWorkflowPhase] = useState<WorkflowPhase>("idle");
  const [job, setJob] = useState<VideoToMotionJob | null>(null);
  const [workflowError, setWorkflowError] = useState<string | null>(null);
  const [workflowErrorOwner, setWorkflowErrorOwner] =
    useState<WorkflowErrorOwner | null>(null);
  const [result, setResult] = useState<MotionResultSummary | null>(null);
  const [setupBusy, setSetupBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const handledImportRequest = useRef<number | null>(null);
  const previewUrl = useRef<string | null>(null);
  const runtimeRequest = useRef<AbortController | null>(null);
  const operation = useRef<AbortController | null>(null);
  const generating = workflowPhase === "uploading" || workflowPhase === "running";
  const busy = generating;

  const refreshRuntime = useCallback((fresh = false) => {
    if (fresh) invalidateGvhmrRuntimeStatus();
    runtimeRequest.current?.abort();
    const request = new AbortController();
    runtimeRequest.current = request;
    setRuntimePhase("checking");
    setRuntimeError(null);
    void getGvhmrRuntimeStatus(request.signal)
      .then((status) => {
        if (request.signal.aborted) return;
        setRuntime(status);
        setRuntimePhase(status.ready ? "ready" : "unavailable");
      })
      .catch((error: unknown) => {
        if (request.signal.aborted) return;
        setRuntime(null);
        setRuntimePhase("error");
        setRuntimeError(errorMessage(error));
      });
  }, []);

  useEffect(() => {
    refreshRuntime();
    return () => runtimeRequest.current?.abort();
  }, [refreshRuntime, runtimeRevision]);

  useEffect(
    () => () => {
      operation.current?.abort();
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    },
    [],
  );

  useEffect(() => {
    if (
      !importRequest ||
      importRequest.target !== "video-file" ||
      handledImportRequest.current === importRequest.id
    ) {
      return;
    }
    handledImportRequest.current = importRequest.id;
    fileInput.current?.click();
  }, [importRequest]);

  const selectVideo = (file: File | null) => {
    if (!file) return;
    if (!isSupportedVideoName(file.name)) {
      setWorkflowPhase("error");
      setWorkflowErrorOwner("selection");
      setWorkflowError(
        text(
          "Supported formats are MP4, MOV, MKV, AVI, WebM, and M4V.",
          "支持 MP4、MOV、MKV、AVI、WebM 和 M4V 格式。",
        ),
      );
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    const previousUrl = previewUrl.current;
    previewUrl.current = nextUrl;
    setVideo({ file, previewUrl: nextUrl, duration: null });
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    setWorkflowPhase("idle");
    setWorkflowError(null);
    setWorkflowErrorOwner(null);
    setJob(null);
    setResult(null);
  };

  const configureRuntime = async () => {
    setSetupBusy(true);
    setRuntimeError(null);
    try {
      const setup = await setupGvhmrInDesktop();
      if (setup.action === "configured") refreshRuntime(true);
    } catch (error) {
      setRuntimePhase("error");
      setRuntimeError(errorMessage(error));
    } finally {
      setSetupBusy(false);
    }
  };

  const run = async () => {
    if (!video || runtimePhase !== "ready") return;
    let parsedFocalLength: number | undefined;
    try {
      parsedFocalLength = parseOptionalFocalLength(focalLength);
    } catch (error) {
      setWorkflowPhase("error");
      setWorkflowErrorOwner("generation");
      setWorkflowError(
        text(errorMessage(error), "焦距必须是正整数。"),
      );
      return;
    }

    operation.current?.abort();
    const request = new AbortController();
    operation.current = request;
    setWorkflowPhase("uploading");
    setWorkflowError(null);
    setWorkflowErrorOwner(null);
    setJob(null);
    setResult(null);
    try {
      const jobId = await startVideoToMotion(
        { video: video.file, staticCamera, focalLength: parsedFocalLength },
        request.signal,
      );
      if (request.signal.aborted) return;
      setWorkflowPhase("running");
      const motion = await waitForVideoToMotion(jobId, {
        signal: request.signal,
        onUpdate: (snapshot) => {
          if (!request.signal.aborted) setJob(snapshot);
        },
      });
      if (request.signal.aborted) return;
      const stageMotion = toStageMotionPayload(motion);
      if (!stageMotion) {
        throw new Error(text("The generated motion has no preview data.", "生成的动作没有预览数据。"));
      }
      setResult(summarizeMotionResult(motion, video.file.name));
      onMotionLoaded?.(stageMotion);
      onMotionLibraryChange?.();
      setWorkflowPhase("done");
    } catch (error) {
      if (request.signal.aborted) return;
      setWorkflowPhase("error");
      setWorkflowErrorOwner("generation");
      setWorkflowError(visibleGvhmrFailure(error, text));
    } finally {
      if (operation.current === request) operation.current = null;
    }
  };

  const runtimeLabel =
    setupBusy
      ? text("Setting up", "配置中")
      : runtimePhase === "checking"
        ? text("Checking", "检查中")
        : runtimePhase === "ready"
          ? `${text("Ready", "就绪")} · ${runtime?.runtime === "docker" ? "Docker" : text("Local", "本地")}`
          : runtimePhase === "unavailable"
            ? text("Unavailable", "不可用")
            : text("Check failed", "检查失败");
  const runtimeDot =
    setupBusy || runtimePhase === "checking"
      ? "bg-primary"
      : runtimePhase === "ready"
        ? "bg-success"
        : runtimePhase === "unavailable"
          ? "bg-warning"
          : "bg-danger";
  const missing = runtime?.missing ?? [];
  const progress = workflowPhase === "uploading" ? 0 : (job?.progress ?? 0);
  const canRun = Boolean(video) && runtimePhase === "ready" && !busy;
  const selectionFailed = workflowErrorOwner === "selection" && Boolean(workflowError);
  const generationFailed = workflowErrorOwner === "generation" && Boolean(workflowError);
  const generationBlocked =
    workflowPhase === "idle" && Boolean(video) && runtimePhase === "unavailable";

  const selectionStatus = selectionFailed
    ? text("Invalid video", "视频无效")
    : video?.file.name ?? text("Not selected", "未选择");
  const selectionTone: WorkflowStatusTone = selectionFailed
    ? "danger"
    : video
      ? "success"
      : "neutral";
  const runtimeTone: WorkflowStatusTone =
    setupBusy || runtimePhase === "checking"
      ? "info"
      : runtimePhase === "ready"
        ? "success"
        : runtimePhase === "unavailable"
          ? "warning"
          : "danger";
  const generationStatus = generationFailed
    ? text("Failed", "失败")
    : generating
      ? `${Math.round(progress * 100)}%`
      : workflowPhase === "done"
        ? text("Done", "完成")
        : generationBlocked
          ? text("Blocked", "已阻塞")
          : text("Waiting", "等待中");
  const generationTone: WorkflowStatusTone = generationFailed
    ? "danger"
    : generating
      ? "info"
      : workflowPhase === "done"
        ? "success"
        : generationBlocked
          ? "warning"
          : "neutral";
  const resultStatus = result
    ? text("Motion Library", "动作资源库")
    : text("Empty", "暂无结果");
  const resultTone: WorkflowStatusTone = result ? "success" : "neutral";

  const pipelineIndex = selectionFailed
    ? 0
    : workflowPhase === "done"
      ? 3
      : generationFailed || generating || (video && runtimePhase === "ready")
        ? 2
        : video
          ? 1
          : 0;
  const pipelineCompletedIndex = selectionFailed
    ? -1
    : generationFailed
      ? 1
      : workflowPhase === "done"
        ? 3
        : pipelineIndex - 1;

  return (
    <InspectorPage title={text("Video → Motion", "视频 → 动作")}>
      <WorkflowPipeline
        label={text("Video to Motion pipeline", "视频转动作流程")}
        steps={pipeline}
        activeIndex={pipelineIndex}
        completedIndex={pipelineCompletedIndex}
      />
      <div className="flex shrink-0 flex-col">
        <WorkflowStep
          title={text("1. Select video", "1. 选择视频")}
          status={selectionStatus}
          statusTone={selectionTone}
          defaultOpen
        >
          <div
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              if (!busy) selectVideo(event.dataTransfer.files[0] ?? null);
            }}
          >
            <ImportDropzone
              label={text("Video import area", "视频导入区域")}
              icon="/icons/sidebar/video-to-motion.svg"
              title={video?.file.name ?? text("Drop a video file here", "将视频文件拖放到这里")}
              hint={
                video
                  ? formatFileSize(video.file.size)
                  : text(
                      "MP4, MOV, MKV, AVI, WebM or M4V",
                      "MP4、MOV、MKV、AVI、WebM 或 M4V",
                    )
              }
            >
              <input
                ref={fileInput}
                className="hidden"
                type="file"
                accept="video/mp4,video/quicktime,video/x-matroska,video/x-msvideo,video/webm,.m4v"
                onChange={(event) => {
                  selectVideo(event.currentTarget.files?.[0] ?? null);
                  event.currentTarget.value = "";
                }}
                disabled={busy}
              />
              <Button size="sm" onClick={() => fileInput.current?.click()} disabled={busy}>
                {text("Choose video", "选择视频")}
              </Button>
            </ImportDropzone>
          </div>
          {video && (
            <section className="mt-3 grid gap-2" aria-label={text("Selected video", "已选择的视频")}>
              <video
                key={video.previewUrl}
                className="aspect-video w-full rounded-md bg-black object-contain"
                src={video.previewUrl}
                controls
                preload="metadata"
                aria-label={text("Selected video preview", "所选视频预览")}
                onLoadedMetadata={(event) => {
                  const duration = Number.isFinite(event.currentTarget.duration)
                    ? event.currentTarget.duration
                    : null;
                  setVideo((current) =>
                    current?.previewUrl === video.previewUrl
                      ? { ...current, duration }
                      : current,
                  );
                }}
              />
              <div className="flex min-w-0 items-center justify-between gap-3">
                <div className="min-w-0 text-[11px] leading-relaxed">
                  <p className="truncate font-semibold text-foreground" title={video.file.name}>
                    {video.file.name}
                  </p>
                  <p className="truncate text-muted-foreground">
                    {[
                      formatFileSize(video.file.size),
                      video.file.type || text("Video", "视频"),
                      video.duration === null ? null : `${video.duration.toFixed(1)} s`,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                </div>
                <Button size="sm" onClick={() => fileInput.current?.click()} disabled={busy}>
                  {text("Replace", "替换")}
                </Button>
              </div>
            </section>
          )}
          {selectionFailed && workflowError && (
            <p className="mt-2.5 rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] leading-relaxed text-danger break-words" role="alert">
              {workflowError}
            </p>
          )}
        </WorkflowStep>

        <WorkflowStep
          title={text("2. Environment", "2. 运行环境")}
          status={runtimeLabel}
          statusTone={runtimeTone}
          defaultOpen
        >
          <div className="grid gap-2.5">
            <div className="flex items-center gap-2 text-xs" role="status" aria-live="polite">
              <span className={`size-2 shrink-0 rounded-full ${runtimeDot}`} aria-hidden="true" />
              <span className="min-w-0 flex-1 text-muted-foreground">
                GVHMR · {text("official weights", "官方权重")}
              </span>
              {canSetupGvhmrInDesktop() && runtimePhase !== "ready" && (
                <Button
                  size="sm"
                  onClick={() => void configureRuntime()}
                  disabled={runtimePhase === "checking" || busy || setupBusy}
                >
                  {setupBusy ? text("Setting up…", "配置中…") : text("Set up", "配置")}
                </Button>
              )}
              <RefreshButton
                label={text("Refresh GVHMR status", "刷新 GVHMR 状态")}
                busy={runtimePhase === "checking"}
                variant="ghost"
                onClick={() => refreshRuntime(true)}
                disabled={busy || setupBusy}
              />
            </div>
            <Field label={text("Weights", "权重")}>
              <select className={fieldClass} defaultValue="official" disabled>
                <option value="official">{text("Official weights", "官方权重")}</option>
              </select>
            </Field>
            {(runtimeError || missing.length > 0) && (
              <div
                className={cn(
                  "rounded-md border px-2.5 py-2 text-[11px] leading-relaxed break-all",
                  runtimeError
                    ? "border-danger-border bg-danger-muted text-danger"
                    : "border-warning-border bg-warning-muted text-warning",
                )}
                role={runtimeError ? "alert" : undefined}
              >
                <p>{runtimeError ?? missing[0]}</p>
                {!runtimeError && missing.length > 1 && (
                  <details className="mt-1">
                    <summary className="w-fit cursor-pointer font-semibold">
                      {text(
                        `${missing.length - 1} more checks`,
                        `另有 ${missing.length - 1} 项检查`,
                      )}
                    </summary>
                    <ul className="mt-1.5 grid list-disc gap-1 pl-4">
                      {missing.slice(1).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
            <SmplxModelLinks runtime={runtime} />
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("3. Generate", "3. 生成动作")}
          status={generationStatus}
          statusTone={generationTone}
          defaultOpen
        >
          <form
            className="grid gap-2.5"
            onSubmit={(event) => {
              event.preventDefault();
              void run();
            }}
          >
            <label className="flex min-h-8 items-center justify-between gap-3 text-xs font-medium text-foreground">
              {text("Static camera", "静态相机")}
              <input
                type="checkbox"
                checked={staticCamera}
                onChange={(event) => setStaticCamera(event.target.checked)}
                disabled={busy}
                className="size-4 accent-primary"
              />
            </label>
            <Field label={text("Focal length", "焦距")}>
              <input
                className={fieldClass}
                type="number"
                inputMode="numeric"
                min="1"
                step="1"
                placeholder={text("Auto", "自动")}
                value={focalLength}
                onChange={(event) => setFocalLength(event.target.value)}
                disabled={busy}
              />
            </Field>
            <Button type="submit" variant="primary" size="sm" disabled={!canRun}>
              {workflowPhase === "uploading"
                ? text("Uploading…", "上传中…")
                : workflowPhase === "running"
                  ? text("Generating…", "生成中…")
                  : text("Start GVHMR", "启动 GVHMR")}
            </Button>
            {generating && (
              <div className="grid gap-1.5 text-[11px] text-muted-foreground" role="status">
                <div className="flex items-center justify-between gap-3">
                  <span className="min-w-0 truncate">
                    {job?.message ?? text("Sending source video", "正在发送源视频")}
                  </span>
                  <strong className="shrink-0 text-foreground">
                    {Math.round(progress * 100)}%
                  </strong>
                </div>
                <progress className="h-1.5 w-full accent-primary" value={progress} max="1" />
              </div>
            )}
            {generationFailed && workflowError && (
              <p className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] leading-relaxed text-danger break-words" role="alert">
                {workflowError}
              </p>
            )}
          </form>
        </WorkflowStep>

        <WorkflowStep
          title={text("4. Motion result", "4. 动作结果")}
          status={resultStatus}
          statusTone={resultTone}
          defaultOpen
        >
          {result ? (
            <div className="grid gap-2">
              <p className="truncate text-xs font-semibold text-foreground" title={result.name}>
                {result.name}
              </p>
              <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md bg-border-subtle text-[11px]">
                {[
                  [text("Frames", "帧数"), formatMetric(result.frames)],
                  [text("Duration", "时长"), formatMetric(result.duration, " s")],
                  [text("Frame rate", "帧率"), formatMetric(result.framerate, " fps")],
                  [text("Library", "资源库"), result.linkedFolder ?? text("Registered", "已登记")],
                ].map(([label, value]) => (
                  <div key={label} className="min-w-0 bg-surface p-2.5">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="mt-0.5 truncate font-semibold text-foreground" title={value}>
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {text(
                "Completed motion will be registered in the Motion Library.",
                "生成完成的动作将登记到动作资源库。",
              )}
            </p>
          )}
        </WorkflowStep>
      </div>
    </InspectorPage>
  );
}
