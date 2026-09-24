import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { Field, fieldClass } from "@/components/Field";
import { InspectorPage } from "@/components/Inspector";
import { Button } from "@/components/ui/button";
import { WorkflowPipeline, WorkflowStep } from "@/components/WorkflowSteps";
import { useLocaleText } from "@/LocaleProvider";
import { getMotionLibrary, loadMotionLibraryEntry } from "@/features/motion/api";
import { getRobotLibrary, loadRobot, type RobotSummary } from "@/features/robot/api";
import { desktopSettingsBridge } from "@/features/settings/api";
import { displayFileName } from "@/lib/api";
import type { StageMotionPayload } from "@/stage/types";

import {
  analyzeDataset,
  computeDatasetSubset,
  downloadBlob,
  exportDatasetManifest,
  exportRobotSubset,
  getCachedDatasetResult,
  getDatasetCatalog,
  motionEntryForAnalysisClip,
  previewDatasetRobot,
  removeDatasetUploadFolder,
  scanDataset,
  uploadDataset,
  type AnalysisRobotPreview,
  type AnalysisEmbedding,
  type DatasetAnalysisResult,
  type DatasetCatalog,
  type DatasetClip,
  type DatasetSummary,
  type DatasetUploadSummary,
  type Histogram,
} from "./api";
import { clipMatchesFilters, selectScatterClip } from "./model";
import { ScatterPlot } from "./ScatterPlot";
import { UploadBasket } from "./UploadBasket";

type BusyAction = "scan" | "upload" | "remove" | "analyze" | "subset" | "preview" | null;

export interface AnalysisViewProps {
  /** App-owned cache policy shared with Workspace Settings. */
  readonly forceAnalysis: boolean;
  /** Optional Stage handoff for human clips selected from the result table. */
  readonly onMotionLoaded?: (motion: StageMotionPayload | null) => void;
  /** Robot previews remain owned by Analysis instead of replacing a workspace robot. */
  readonly onRobotPreviewLoaded?: (preview: AnalysisRobotPreview | null) => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function numberValue(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(value: unknown): string {
  const parsed = numberValue(value);
  if (parsed === null) return "-";
  if (Math.abs(parsed) >= 100) return parsed.toFixed(1);
  if (Math.abs(parsed) >= 10) return parsed.toFixed(2);
  return parsed.toFixed(3);
}

function clipLabel(clip: DatasetClip, fallback = "Clip"): string {
  return clip.clip_id || displayFileName(clip.source_path, fallback);
}

function validClips(result: DatasetAnalysisResult | null): DatasetClip[] {
  return result?.clips.filter(
    (clip) => !clip.error && Object.keys(clip.metrics).length > 0,
  ) ?? [];
}

function HistogramChart({
  histogram,
  range,
  onRangeChange,
}: {
  histogram: Histogram | undefined;
  range: readonly [number, number] | null;
  onRangeChange: (range: readonly [number, number] | null) => void;
}) {
  const text = useLocaleText();
  const drag = useRef<{ pointerId: number; start: number } | null>(null);
  if (!histogram) {
    return <p className="text-xs text-muted-foreground">{text("No values for this metric.", "此指标没有可用数值。")}</p>;
  }
  const max = Math.max(...histogram.counts, 1);
  const width = 420;
  const height = 116;
  const gap = 2;
  const barWidth = width / Math.max(histogram.counts.length, 1);

  function binAt(event: ReactPointerEvent<SVGSVGElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    return Math.max(0, Math.min(histogram!.counts.length - 1, Math.floor(x / barWidth)));
  }

  function applyBins(first: number, last: number): void {
    const low = Math.min(first, last);
    const high = Math.max(first, last);
    const lo = histogram!.edges[low];
    const hi = histogram!.edges[high + 1];
    if (Number.isFinite(lo) && Number.isFinite(hi)) onRangeChange([lo, hi]);
  }

  return (
    <div className="grid gap-1.5">
      <svg
        className="h-[116px] w-full touch-none overflow-visible"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label={text("Metric histogram", "指标直方图")}
        onPointerDown={(event) => {
          if (event.button !== 0 || !histogram.counts.length) return;
          const start = binAt(event);
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { pointerId: event.pointerId, start };
          applyBins(start, start);
        }}
        onPointerMove={(event) => {
          if (drag.current?.pointerId !== event.pointerId) return;
          applyBins(drag.current.start, binAt(event));
        }}
        onPointerUp={(event) => {
          if (drag.current?.pointerId !== event.pointerId) return;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        {histogram.counts.map((count, index) => {
          const barHeight = (count / max) * (height - 18);
          const selected = Boolean(
            range &&
            histogram.edges[index + 1] >= range[0] &&
            histogram.edges[index] <= range[1],
          );
          return (
            <rect
              key={index}
              x={index * barWidth + gap / 2}
              y={height - barHeight - 1}
              width={Math.max(1, barWidth - gap)}
              height={barHeight}
              rx="2"
              className={!range ? "fill-primary/75" : selected ? "fill-primary" : "fill-primary/25"}
              stroke={selected ? "currentColor" : "none"}
              strokeWidth={selected ? 1 : 0}
              role="button"
              tabIndex={0}
              aria-label={`${formatNumber(histogram.edges[index])} ${text("to", "至")} ${formatNumber(histogram.edges[index + 1])}: ${count} ${text("clips", "个片段")}`}
              aria-pressed={selected}
              onKeyDown={(event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                applyBins(index, index);
              }}
            />
          );
        })}
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} className="stroke-border" />
      </svg>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{formatNumber(histogram.min)}</span>
        {range ? (
          <button type="button" className="rounded-sm px-1 text-primary hover:bg-accent" onClick={() => onRangeChange(null)}>
            {formatNumber(range[0])}–{formatNumber(range[1])} · {text("Clear", "清除")}
          </button>
        ) : (
          <span>{text("mean", "均值")} {formatNumber(histogram.mean)}</span>
        )}
        <span>{formatNumber(histogram.max)}</span>
      </div>
    </div>
  );
}

function SummaryCards({ summary }: { summary: DatasetSummary }) {
  const text = useLocaleText();
  const cards = [
    [text("Clips", "片段"), summary.num_clips],
    [text("Analyzed", "已分析"), summary.num_ok],
    [text("Failed", "失败"), summary.num_error],
    [text("Clusters", "聚类"), Object.keys(summary.cluster_counts).length],
  ] as const;
  return (
    <div className="grid grid-cols-4 gap-1.5">
      {cards.map(([label, value]) => (
        <div key={label} className="rounded-md border border-border-subtle bg-surface px-2.5 py-2">
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</p>
          <p className="mt-0.5 text-lg font-semibold text-foreground">{value}</p>
        </div>
      ))}
    </div>
  );
}

export function AnalysisView({
  forceAnalysis,
  onMotionLoaded,
  onRobotPreviewLoaded,
}: AnalysisViewProps) {
  const text = useLocaleText();
  const pipeline = [
    text("Select Data", "选择数据"),
    text("Configure", "配置"),
    text("Analyze", "分析"),
    text("Results", "结果"),
  ];
  const analysisClipLabel = (clip: DatasetClip) =>
    clipLabel(clip, text("Clip", "片段"));
  const [catalog, setCatalog] = useState<DatasetCatalog>({});
  const [robots, setRobots] = useState<readonly RobotSummary[]>([]);
  const [source, setSource] = useState("");
  const [defaultSource, setDefaultSource] = useState("");
  const [sourceSummary, setSourceSummary] = useState<DatasetUploadSummary | null>(null);
  const [uploadSource, setUploadSource] = useState<string | null>(null);
  const [embedding, setEmbedding] = useState<AnalysisEmbedding>("handcrafted");
  const [result, setResult] = useState<DatasetAnalysisResult | null>(null);
  const [busy, setBusy] = useState<BusyAction>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [tagFilter, setTagFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [folderFilter, setFolderFilter] = useState("all");
  const [metric, setMetric] = useState("complexity");
  const [metricRange, setMetricRange] = useState<readonly [number, number] | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [subsetIds, setSubsetIds] = useState<Set<string>>(new Set());
  const [subsetRatio, setSubsetRatio] = useState("10");
  const [subsetAlpha, setSubsetAlpha] = useState("0.99");
  const [previewing, setPreviewing] = useState<string | null>(null);
  const [previewRobot, setPreviewRobot] = useState("");
  const folderInput = useRef<HTMLInputElement | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const request = new AbortController();
    requestRef.current = request;
    void Promise.all([
      getDatasetCatalog({ signal: request.signal }),
      getMotionLibrary({ signal: request.signal }),
      getRobotLibrary({ signal: request.signal }),
    ])
      .then(([loadedCatalog, library, robotLibrary]) => {
        if (request.signal.aborted) return;
        setCatalog(loadedCatalog);
        setSource(library.source_root);
        setDefaultSource(library.source_root);
        setRobots(robotLibrary.robots.filter((robot) => robot.has_urdf));
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) setError(errorMessage(reason));
      });
    return () => {
      request.abort();
      requestRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!result) return;
    setMetric(result.summary.numeric_keys[0] || "complexity");
    setTagFilter("all");
    setKindFilter("all");
    setFolderFilter("all");
    setMetricRange(null);
    setSelectedIds(new Set());
    setSubsetIds(new Set());
  }, [result]);

  const availableClips = useMemo(() => validClips(result), [result]);
  const folders = useMemo(
    () => [...new Set(availableClips.map((clip) => clip.folder_label).filter(Boolean))].sort(),
    [availableClips],
  );
  const filteredClips = useMemo(
    () =>
      availableClips.filter((clip) =>
        clipMatchesFilters(clip, {
          tag: tagFilter,
          kind: kindFilter,
          folder: folderFilter,
          metric,
          metricRange,
        }),
      ),
    [availableClips, folderFilter, kindFilter, metric, metricRange, tagFilter],
  );
  const visibleIds = useMemo(
    () => new Set(filteredClips.map((clip) => clip.clip_id)),
    [filteredClips],
  );
  const exportIds = useMemo(() => {
    const combined = new Set([...subsetIds, ...selectedIds]);
    if (combined.size) return [...combined];
    return filteredClips.map((clip) => clip.clip_id);
  }, [filteredClips, selectedIds, subsetIds]);
  const selectedRobotCount = useMemo(
    () =>
      exportIds.filter(
        (id) => result?.clips.find((clip) => clip.clip_id === id)?.source_kind === "robot",
      ).length,
    [exportIds, result],
  );
  const hasRobotClips = availableClips.some(
    (clip) => clip.source_kind === "robot" || clip.dataset === "robot",
  );
  const catalogMetric = (catalog.metrics?.[metric] ?? {}) as Record<string, unknown>;
  const summary = result?.summary;
  const tags = summary?.tag_order ?? [];
  const kinds = [...new Set(availableClips.map((clip) => clip.source_kind))].sort();

  function begin(action: Exclude<BusyAction, null>): AbortController {
    requestRef.current?.abort();
    const request = new AbortController();
    requestRef.current = request;
    setBusy(action);
    setProgress(0);
    setError(null);
    return request;
  }

  function finish(request: AbortController): void {
    if (requestRef.current !== request) return;
    requestRef.current = null;
    setBusy(null);
  }

  function clearAnalysisOutput(): void {
    setResult(null);
    setMetricRange(null);
    setSelectedIds(new Set());
    setSubsetIds(new Set());
    setPreviewing(null);
    onMotionLoaded?.(null);
    onRobotPreviewLoaded?.(null);
  }

  function applyUploadSummary(value: DatasetUploadSummary): void {
    setSource(value.source);
    setUploadSource(value.source || null);
    setSourceSummary(value.clip_count ? value : null);
    clearAnalysisOutput();
  }

  async function scanPath(path: string): Promise<void> {
    if (!path.trim()) return;
    const request = begin("scan");
    setSource(path.trim());
    setStatus(text("Scanning dataset...", "正在扫描数据集……"));
    try {
      const value = await scanDataset(path.trim(), { signal: request.signal });
      if (request.signal.aborted) return;
      setSource(value.source);
      setSourceSummary(value);
      setUploadSource(null);
      clearAnalysisOutput();
      setStatus(text(`${value.clip_count} clips found.`, `找到 ${value.clip_count} 个片段。`));
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      finish(request);
    }
  }

  async function chooseFolder(): Promise<void> {
    const desktop = desktopSettingsBridge();
    if (!desktop) {
      folderInput.current?.click();
      return;
    }
    try {
      const selected = await desktop.selectDirectory();
      if (selected) await scanPath(selected);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  }

  function uploadFolder(files: FileList | null): void {
    const selected = files ? Array.from(files) : [];
    if (!selected.length) return;
    const request = begin("upload");
    setStatus(text(`Uploading ${selected.length} files...`, `正在上传 ${selected.length} 个文件……`));
    void uploadDataset(selected, {
      appendTo: uploadSource ?? undefined,
      signal: request.signal,
    })
      .then((value) => {
        if (request.signal.aborted) return;
        applyUploadSummary(value);
        setStatus(text(`${value.clip_count} clips ready for analysis.`, `${value.clip_count} 个片段可供分析。`));
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) setError(errorMessage(reason));
      })
      .finally(() => finish(request));
  }

  async function removeUploadedFolder(folder: string): Promise<void> {
    if (!uploadSource) return;
    const request = begin("remove");
    setStatus(text(`Removing ${folder}...`, `正在移除 ${folder}……`));
    try {
      const value = await removeDatasetUploadFolder(uploadSource, folder, {
        signal: request.signal,
      });
      if (request.signal.aborted) return;
      applyUploadSummary(value);
      setStatus(
        value.clip_count
          ? text(
              `Removed ${folder}. ${value.clip_count} clips remain.`,
              `已移除 ${folder}，剩余 ${value.clip_count} 个片段。`,
            )
          : text(
              `Removed ${folder}. Upload basket is empty.`,
              `已移除 ${folder}，上传篮已清空。`,
            ),
      );
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      finish(request);
    }
  }

  async function clearUploadBasket(): Promise<void> {
    if (!uploadSource || !sourceSummary) return;
    const request = begin("remove");
    const folders = Object.keys(sourceSummary.folders);
    let currentSource = uploadSource;
    setStatus(text("Clearing upload basket...", "正在清空上传篮……"));
    try {
      for (const folder of folders) {
        const value = await removeDatasetUploadFolder(currentSource, folder, {
          signal: request.signal,
        });
        if (request.signal.aborted) return;
        applyUploadSummary(value);
        if (value.source) currentSource = value.source;
      }
      setStatus(text("Upload basket cleared.", "上传篮已清空。"));
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      finish(request);
    }
  }

  function runAnalysis(): void {
    const request = begin("analyze");
    setStatus(text("Analyzing dataset...", "正在分析数据集……"));
    void analyzeDataset(
      {
        ...(source.trim() ? { source: source.trim() } : {}),
        embedding,
        force: forceAnalysis,
      },
      {
        signal: request.signal,
        onUpdate: (job) => {
          if (!request.signal.aborted) {
            setProgress(job.progress ?? 0);
            setStatus(job.message || text("Analyzing dataset...", "正在分析数据集……"));
          }
        },
      },
    )
      .then((value) => {
        if (request.signal.aborted) return;
        setResult(value);
        setProgress(1);
        setStatus(text(
          `Analysis complete: ${value.summary.num_ok} clips.`,
          `分析完成：${value.summary.num_ok} 个片段。`,
        ));
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) setError(errorMessage(reason));
      })
      .finally(() => finish(request));
  }

  async function loadCached(): Promise<void> {
    if (!source.trim()) return;
    const request = begin("scan");
    setStatus(text("Checking cached result...", "正在检查缓存结果……"));
    try {
      const cached = await getCachedDatasetResult(source.trim(), embedding, {
        signal: request.signal,
      });
      if (request.signal.aborted) return;
      if (!cached.available || !cached.clips || !cached.summary || !cached.meta) {
        setStatus(text("No cached result for this source.", "此数据源没有缓存结果。"));
        return;
      }
      setResult(cached as DatasetAnalysisResult);
      setStatus(text("Loaded cached result.", "已加载缓存结果。"));
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      finish(request);
    }
  }

  function recommendSubset(): void {
    if (!result || !filteredClips.length) return;
    const request = begin("subset");
    const ratio = Math.max(1, Math.min(100, Number(subsetRatio) || 10)) / 100;
    const alpha = Math.max(0, Math.min(1, Number(subsetAlpha) || 0.99));
    const k = Math.max(1, Math.round(filteredClips.length * ratio));
    void computeDatasetSubset(filteredClips, k, alpha, { signal: request.signal })
      .then((value) => {
        if (!request.signal.aborted) {
          const recommended = new Set(value.selected);
          setSubsetIds(recommended);
          setSelectedIds((current) =>
            new Set([...current].filter((id) => !recommended.has(id))),
          );
          setStatus(text(`Recommended ${value.count} clips.`, `已推荐 ${value.count} 个片段。`));
        }
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) setError(errorMessage(reason));
      })
      .finally(() => finish(request));
  }

  function toggleSelected(id: string): void {
    setSelectedIds((current) => {
      if (subsetIds.has(id)) return current;
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function activateScatterClip(clip: DatasetClip, additive: boolean): void {
    setSelectedIds((current) =>
      selectScatterClip(current, clip.clip_id, additive, subsetIds),
    );
    void previewClip(clip);
  }

  async function exportManifest(format: "json" | "csv"): Promise<void> {
    if (!result || !exportIds.length) return;
    setError(null);
    try {
      const blob = await exportDatasetManifest({
        clips: result.clips,
        ids: exportIds,
        analyze_source: result.meta.source_root,
        format,
      });
      downloadBlob(blob, `dataset_manifest.${format}`);
      setStatus(text(`Exported ${exportIds.length} clips.`, `已导出 ${exportIds.length} 个片段。`));
    } catch (reason) {
      setError(errorMessage(reason));
    }
  }

  async function exportRobots(): Promise<void> {
    if (!result || !selectedRobotCount) return;
    setError(null);
    try {
      const blob = await exportRobotSubset({ clips: result.clips, ids: exportIds });
      downloadBlob(blob, "robot_subset_export.zip");
      setStatus(text(
        `Exported ${selectedRobotCount} robot clips.`,
        `已导出 ${selectedRobotCount} 个机器人片段。`,
      ));
    } catch (reason) {
      setError(errorMessage(reason));
    }
  }

  async function previewClip(clip: DatasetClip): Promise<void> {
    if (clip.source_kind === "robot" || clip.dataset === "robot") {
      const request = begin("preview");
      setPreviewing(clip.clip_id);
      setStatus(text(
        `Loading robot trajectory ${analysisClipLabel(clip)}...`,
        `正在加载机器人轨迹 ${analysisClipLabel(clip)}……`,
      ));
      try {
        const inferred = typeof clip.metrics.robot_preset === "string"
          ? clip.metrics.robot_preset.trim()
          : "";
        const result = await previewDatasetRobot(
          {
            source_path: clip.source_path,
            ...(previewRobot || inferred ? { robot: previewRobot || inferred } : {}),
          },
          {
            signal: request.signal,
            onUpdate: (job) => {
              if (!request.signal.aborted) {
                setStatus(job.message || text("Loading robot trajectory...", "正在加载机器人轨迹……"));
              }
            },
          },
        );
        if (request.signal.aborted) return;
        const robot = await loadRobot(result.robot, { signal: request.signal });
        if (request.signal.aborted) return;
        onRobotPreviewLoaded?.({
          robot,
          trajectory: result.trajectory,
          scene: result.scaled_scene,
          previewToken: result.preview_token,
        });
        setStatus(text(`Previewing ${result.name}.`, `正在预览 ${result.name}。`));
      } catch (reason) {
        if (!request.signal.aborted) setError(errorMessage(reason));
      } finally {
        if (requestRef.current === request) setPreviewing(null);
        finish(request);
      }
      return;
    }
    const request = begin("preview");
    setPreviewing(clip.clip_id);
    setStatus(text(
      `Loading ${analysisClipLabel(clip)}...`,
      `正在加载 ${analysisClipLabel(clip)}……`,
    ));
    try {
      const motion = await loadMotionLibraryEntry(motionEntryForAnalysisClip(clip), {
        signal: request.signal,
        onUpdate: (job) => {
          if (!request.signal.aborted) setStatus(job.message || text("Loading clip...", "正在加载片段……"));
        },
      });
      if (!request.signal.aborted) {
        onRobotPreviewLoaded?.(null);
        onMotionLoaded?.(motion);
        setStatus(text(
          `Previewing ${analysisClipLabel(clip)}.`,
          `正在预览 ${analysisClipLabel(clip)}。`,
        ));
      }
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (requestRef.current === request) setPreviewing(null);
      finish(request);
    }
  }

  return (
    <InspectorPage title={text("Data Analysis", "数据分析")}>
      <WorkflowPipeline
        label={text("Data Analysis pipeline", "数据分析流程")}
        steps={pipeline}
        activeIndex={result ? 3 : busy === "analyze" ? 2 : sourceSummary ? 1 : 0}
        completedIndex={
          result ? 3 : busy === "analyze" ? 1 : sourceSummary ? 0 : -1
        }
      />

      <div className="flex shrink-0 flex-col">
        <WorkflowStep
          title={text("1. Select data", "1. 选择数据")}
          status={sourceSummary
            ? text(`${sourceSummary.clip_count} clips`, `${sourceSummary.clip_count} 个片段`)
            : text("No data", "无数据")}
          defaultOpen
        >
          <div className="grid gap-2.5">
            <div className="grid grid-cols-2 gap-2">
              <Button size="sm" disabled={busy !== null} onClick={() => void chooseFolder()}>
                {text("Choose folder", "选择文件夹")}
              </Button>
              <Button
                size="sm"
                disabled={!defaultSource || busy !== null}
                onClick={() => {
                  void scanPath(defaultSource);
                }}
              >
                {text("Built-in library", "内置资源库")}
              </Button>
            </div>
            <input
              ref={folderInput}
              className="hidden"
              type="file"
              multiple
              {...({ webkitdirectory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
              onChange={(event) => {
                uploadFolder(event.currentTarget.files);
                event.currentTarget.value = "";
              }}
            />
            {sourceSummary && (
              <p className="text-xs text-muted-foreground">
                {sourceSummary.human_count} {text("human", "人体")} · {sourceSummary.robot_count} {text("robot", "机器人")} · {Object.keys(sourceSummary.folders).length} {text("folders", "个文件夹")}
              </p>
            )}
            {sourceSummary && uploadSource && (
              <UploadBasket
                summary={sourceSummary}
                disabled={busy !== null}
                onRemove={(folder) => void removeUploadedFolder(folder)}
                onClear={() => void clearUploadBasket()}
              />
            )}
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("2. Configure", "2. 配置")}
          status={embedding === "handcrafted"
            ? text("Handcrafted", "手工特征")
            : text("Reserved", "预留")}
        >
          <div className="grid gap-2.5">
            <Field label={text("Embedding", "嵌入方法")}>
              <select
                className={fieldClass}
                value={embedding}
                disabled={busy !== null}
                onChange={(event) => setEmbedding(event.target.value as AnalysisEmbedding)}
              >
                <option value="handcrafted">{text("Handcrafted features", "手工特征")}</option>
                <option value="pae" disabled>{text("PAE (reserved)", "PAE（预留）")}</option>
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="primary"
                size="sm"
                disabled={!source.trim() || busy !== null}
                onClick={runAnalysis}
              >
                {busy === "analyze" ? text("Analyzing...", "分析中……") : text("Start analysis", "开始分析")}
              </Button>
              <Button size="sm" disabled={!source.trim() || busy !== null} onClick={() => void loadCached()}>
                {text("Load existing result", "加载已有结果")}
              </Button>
            </div>
            {busy === "analyze" && (
              <div
                className="h-1.5 overflow-hidden rounded-full bg-border-subtle"
                role="progressbar"
                aria-valuenow={Math.round(progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div className="h-full bg-primary transition-[width]" style={{ width: `${Math.max(2, progress * 100)}%` }} />
              </div>
            )}
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("3. Analyze", "3. 分析")}
          status={busy === "analyze"
            ? text("Running", "运行中")
            : result
              ? text("Complete", "已完成")
              : text("Not started", "未开始")}
        >
          <div className="grid gap-2">
            <p className="text-xs text-muted-foreground">
              {status || text(
                "Run analysis to calculate dynamics, quality, tags, embedding, and clusters.",
                "运行分析以计算动力学、质量、标签、嵌入和聚类。",
              )}
            </p>
            {error && (
              <p className="break-words rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] text-danger" role="alert">
                {error}
              </p>
            )}
          </div>
        </WorkflowStep>

        <WorkflowStep
          title={text("4. Results", "4. 结果")}
          status={summary
            ? text(`${summary.num_ok} analyzed`, `已分析 ${summary.num_ok} 个`)
            : text("No results", "无结果")}
          defaultOpen={Boolean(result)}
        >
          {!result || !summary ? (
            <p className="text-xs leading-[1.5] text-muted-foreground">{text("Analysis results will appear here.", "分析结果将显示在这里。")}</p>
          ) : (
            <div className="grid gap-3">
              <SummaryCards summary={summary} />

              {hasRobotClips && (
                <Field label={text("Preview robot", "预览机器人")}>
                  <select
                    className={fieldClass}
                    value={previewRobot}
                    disabled={busy !== null}
                    onChange={(event) => setPreviewRobot(event.target.value)}
                  >
                    <option value="">{text("Auto-detect from trajectory", "从轨迹自动识别")}</option>
                    {robots.map((robot) => (
                      <option key={robot.name} value={robot.name}>
                        {robot.display_name}
                      </option>
                    ))}
                  </select>
                </Field>
              )}

              {summary.num_error > 0 && (
                <details className="rounded-md border border-warning-border bg-warning-muted px-2.5 py-2">
                  <summary className="cursor-pointer list-none text-xs font-semibold text-warning [&::-webkit-details-marker]:hidden">
                    {text(
                      `${summary.num_error} clips could not be analyzed`,
                      `${summary.num_error} 个片段无法分析`,
                    )}
                  </summary>
                  <div className="mt-2 grid gap-1.5">
                    {result.clips
                      .filter((clip) => clip.error)
                      .map((clip) => (
                        <div key={clip.clip_id} className="grid gap-0.5 text-[11px] text-warning">
                          <span className="font-medium">{analysisClipLabel(clip)}</span>
                          <span className="break-words opacity-80">{clip.error}</span>
                        </div>
                      ))}
                  </div>
                </details>
              )}

              <div className="grid gap-2 rounded-md border border-border-subtle bg-background p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold text-foreground">{text("Embedding map", "嵌入图")}</h2>
                  <span className="text-[11px] text-muted-foreground">
                    {text(
                      `${Object.keys(summary.cluster_counts).length} clusters`,
                      `${Object.keys(summary.cluster_counts).length} 个聚类`,
                    )}
                  </span>
                </div>
                <ScatterPlot
                  clips={availableClips}
                  visibleIds={visibleIds}
                  selectedIds={selectedIds}
                  subsetIds={subsetIds}
                  onActivate={activateScatterClip}
                />
              </div>

              <div className="grid gap-2 rounded-md border border-border-subtle bg-background p-2.5">
                <div className="grid grid-cols-3 gap-1.5">
                  <select className={fieldClass} aria-label={text("Analysis tag filter", "分析标签筛选")} value={tagFilter} onChange={(event) => setTagFilter(event.target.value)}>
                    <option value="all">{text("All tags", "全部标签")}</option>
                    {tags.map((tag) => <option key={tag} value={tag}>{tag} ({summary.tag_counts[tag] ?? 0})</option>)}
                  </select>
                  <select className={fieldClass} aria-label={text("Analysis source kind filter", "分析来源筛选")} value={kindFilter} onChange={(event) => setKindFilter(event.target.value)}>
                    <option value="all">{text("All sources", "全部来源")}</option>
                    {kinds.map((kind) => (
                      <option key={kind} value={kind}>
                        {kind === "human"
                          ? text("Human", "人体")
                          : kind === "robot"
                            ? text("Robot", "机器人")
                            : kind}
                      </option>
                    ))}
                  </select>
                  <select className={fieldClass} aria-label={text("Analysis folder filter", "分析文件夹筛选")} value={folderFilter} onChange={(event) => setFolderFilter(event.target.value)}>
                    <option value="all">{text("All folders", "全部文件夹")}</option>
                    {folders.map((folder) => <option key={folder} value={folder}>{folder}</option>)}
                  </select>
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
                  <div>
                    <select
                      className={fieldClass}
                      aria-label={text("Analysis metric", "分析指标")}
                      value={metric}
                      onChange={(event) => {
                        setMetric(event.target.value);
                        setMetricRange(null);
                      }}
                    >
                      {summary.numeric_keys.map((key) => <option key={key} value={key}>{key}</option>)}
                    </select>
                    {typeof catalogMetric.desc === "string" && <p className="mt-1 text-[11px] text-muted-foreground">{catalogMetric.desc}</p>}
                  </div>
                  <div className="min-w-[120px] text-right text-[11px] text-muted-foreground">
                    <p>{text("median", "中位数")} {formatNumber(summary.histograms[metric]?.median)}</p>
                    <p>{text("mean", "均值")} {formatNumber(summary.histograms[metric]?.mean)}</p>
                  </div>
                </div>
                <HistogramChart
                  histogram={summary.histograms[metric]}
                  range={metricRange}
                  onRangeChange={setMetricRange}
                />
              </div>

              <div className="grid gap-2 rounded-md border border-border-subtle bg-background p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold text-foreground">{text("Recommended subset", "推荐子集")}</h2>
                  <span className="text-[11px] text-muted-foreground">
                    {subsetIds.size
                      ? text(`${subsetIds.size} recommended`, `已推荐 ${subsetIds.size} 个`)
                      : text(`${exportIds.length} selected for export`, `已选择 ${exportIds.length} 个用于导出`)}
                  </span>
                </div>
                <div className="grid grid-cols-[1fr_1fr_auto] items-end gap-2">
                  <Field label={text("Ratio %", "比例 %")}><input className={fieldClass} type="number" min="1" max="100" value={subsetRatio} onChange={(event) => setSubsetRatio(event.target.value)} /></Field>
                  <Field label={text("Coverage alpha", "覆盖率 alpha")}><input className={fieldClass} type="number" min="0" max="1" step="0.01" value={subsetAlpha} onChange={(event) => setSubsetAlpha(event.target.value)} /></Field>
                  <Button size="sm" disabled={busy !== null || !filteredClips.length} onClick={recommendSubset}>{busy === "subset" ? text("Selecting...", "选择中……") : text("Recommend", "推荐")}</Button>
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  <Button
                    size="sm"
                    disabled={busy !== null || !filteredClips.length}
                    onClick={() => setSelectedIds(new Set(
                      filteredClips
                        .map((clip) => clip.clip_id)
                        .filter((id) => !subsetIds.has(id)),
                    ))}
                  >
                    {text("Select visible", "选择可见项")}
                  </Button>
                  <Button size="sm" disabled={busy !== null || !selectedIds.size} onClick={() => setSelectedIds(new Set())}>{text("Clear selection", "清除选择")}</Button>
                  <Button size="sm" disabled={busy !== null || !exportIds.length} onClick={() => void exportManifest("json")}>{text("Export JSON", "导出 JSON")}</Button>
                  <Button size="sm" disabled={busy !== null || !exportIds.length} onClick={() => void exportManifest("csv")}>{text("Export CSV", "导出 CSV")}</Button>
                </div>
                <Button size="sm" disabled={busy !== null || !selectedRobotCount} onClick={() => void exportRobots()}>{text("Export robot ZIP", "导出机器人 ZIP")} ({selectedRobotCount})</Button>
              </div>

              <div className="grid gap-1.5">
                <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                  <span>{text(`${filteredClips.length} visible clips`, `${filteredClips.length} 个可见片段`)}</span>
                  <span>{text(`${selectedIds.size} manually selected`, `手动选择 ${selectedIds.size} 个`)}</span>
                </div>
                <div className="grid max-h-[300px] gap-1 overflow-y-auto pr-1">
                  {filteredClips.map((clip) => {
                    const recommended = subsetIds.has(clip.clip_id);
                    return (
                      <div key={clip.clip_id} className="grid grid-cols-[auto_minmax(0,1fr)_72px_auto] items-center gap-2 rounded-md border border-border-subtle bg-surface px-2 py-1.5">
                        <input type="checkbox" className="size-4 accent-primary" checked={selectedIds.has(clip.clip_id)} onChange={() => toggleSelected(clip.clip_id)} aria-label={text(`Select ${analysisClipLabel(clip)}`, `选择 ${analysisClipLabel(clip)}`)} />
                        <button type="button" className="min-w-0 truncate text-left text-xs font-medium text-foreground hover:text-primary" title={analysisClipLabel(clip)} onClick={() => toggleSelected(clip.clip_id)}>
                          {analysisClipLabel(clip)}
                          <span className="ml-1 text-[10px] font-normal text-muted-foreground">{clip.folder_label}</span>
                        </button>
                        <span className="text-right text-[11px] text-muted-foreground">{formatNumber(clip.metrics[metric])}</span>
                        <div className="flex items-center gap-1">
                          {recommended && <span className="text-[10px] font-semibold text-warning">FPS</span>}
                          <Button size="sm" disabled={busy !== null} onClick={() => void previewClip(clip)}>{previewing === clip.clip_id ? "..." : text("Preview", "预览")}</Button>
                        </div>
                      </div>
                    );
                  })}
                  {!filteredClips.length && <p className="py-4 text-center text-xs text-muted-foreground">{text("No clips match these filters.", "没有片段符合当前筛选条件。")}</p>}
                </div>
              </div>
            </div>
          )}
        </WorkflowStep>
      </div>
    </InspectorPage>
  );
}
