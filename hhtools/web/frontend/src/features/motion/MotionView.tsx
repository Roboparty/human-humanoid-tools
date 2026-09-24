import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ImportDropzone } from "@/components/ImportDropzone";
import { InspectorPage } from "@/components/Inspector";
import { RefreshButton } from "@/components/RefreshButton";
import { SearchField } from "@/components/SearchField";
import { ValidationSummary } from "@/components/ValidationSummary";
import { motionValidationFacts } from "@/components/validationFacts";
import { Button } from "@/components/ui/button";
import type { ApplicationImportRequest } from "@/importIntent";
import { useLocaleText } from "@/LocaleProvider";
import { displayFileName } from "@/lib/api";
import { SegmentedControl } from "@/components/SegmentedControl";
import type { StageMotionPayload } from "@/stage/types";

import {
  getHumanMotionLibrary,
  loadMotionLibraryEntry,
  toStageMotionPayload,
  uploadMotion,
  type MotionCategory,
  type MotionLibraryEntry,
  type MotionProfile,
} from "./api";

interface MotionProfileOption {
  id: MotionProfile;
  label: string;
  prompt: string;
  promptZh: string;
  icon: string;
  acceptsFile: boolean;
}

const profiles: readonly MotionProfileOption[] = [
  {
    id: "mimic",
    label: "mimic",
    prompt: "Drop a motion file or folder",
    promptZh: "拖入动作文件或文件夹",
    icon: "/icons/motion/film.svg",
    acceptsFile: true,
  },
  {
    id: "intermimic",
    label: "intermimic",
    prompt: "Drop an object-interaction motion folder",
    promptZh: "拖入物体交互动作文件夹",
    icon: "/icons/motion/package.svg",
    acceptsFile: false,
  },
  {
    id: "meshmimic",
    label: "meshmimic",
    prompt: "Drop a terrain-motion folder",
    promptZh: "拖入地形动作文件夹",
    icon: "/icons/motion/mountain.svg",
    acceptsFile: false,
  },
];

const categories: readonly {
  value: "all" | MotionCategory;
  label: string;
  labelZh: string;
}[] = [
  { value: "all", label: "All", labelZh: "全部" },
  { value: "motion", label: "Motion", labelZh: "动作" },
  { value: "object", label: "Object interaction", labelZh: "物体交互" },
  { value: "terrain", label: "Terrain scene", labelZh: "地形场景" },
];

const categoryBadgeClass: Readonly<Record<MotionCategory, string>> = {
  motion: "bg-[#0071e3]/[0.12] text-[#0071e3]",
  object: "bg-[#8e44ad]/[0.14] text-[#8e44ad]",
  terrain: "bg-[#34c759]/[0.14] text-[#34c759]",
};

const fieldClass =
  "min-h-[30px] min-w-0 truncate rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-foreground disabled:cursor-not-allowed disabled:text-muted-foreground";

function entryKey(entry: MotionLibraryEntry): string {
  return (
    entry.source_path ||
    [entry.folder_label, entry.sequence_id, entry.stem].join("/")
  );
}

function entryCategory(entry: MotionLibraryEntry): MotionCategory {
  return entry.motion_category === "object" || entry.motion_category === "terrain"
    ? entry.motion_category
    : "motion";
}

function entryLabel(entry: MotionLibraryEntry, fallback = "Motion"): string {
  return (
    entry.stem ||
    entry.sequence_id ||
    entry.label ||
    displayFileName(entry.source_path, fallback)
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function uploadFolderLabel(files: readonly File[]): string | undefined {
  const relative = (files[0] as File & { webkitRelativePath?: string })
    ?.webkitRelativePath;
  const label = relative?.split("/")[0]?.trim();
  return label || undefined;
}

export function MotionView({
  currentMotion,
  onMotionLoaded,
  onOpenSettings,
  importRequest,
  libraryRevision = 0,
}: {
  /** App-owned stable input; failed replacements leave it untouched. */
  currentMotion?: StageMotionPayload | null;
  /** App publishes this payload to the shared R3F Stage. */
  onMotionLoaded?: (motion: StageMotionPayload | null) => void;
  /** Directory ownership stays in Workspace Settings. */
  onOpenSettings?: () => void;
  /** App-owned File-menu intent; this mounted view owns its input elements. */
  importRequest?: ApplicationImportRequest | null;
  /** Settings increments this after changing the process-wide library root. */
  libraryRevision?: number;
}) {
  const text = useLocaleText();
  const [profile, setProfile] = useState<MotionProfile>("mimic");
  const [entries, setEntries] = useState<readonly MotionLibraryEntry[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<"all" | MotionCategory>("all");
  const [loadingLibrary, setLoadingLibrary] = useState(true);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const libraryRequest = useRef<AbortController | null>(null);
  const motionRequest = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const folderInput = useRef<HTMLInputElement | null>(null);
  const handledImportRequest = useRef<number | null>(null);
  const selected = profiles.find((item) => item.id === profile) ?? profiles[0];
  const selectedKey = currentMotion?.library_entry?.source_path ?? null;
  const motionLabel = useCallback(
    (entry: MotionLibraryEntry) => entryLabel(entry, text("Motion", "动作")),
    [text],
  );

  const refreshLibrary = useCallback(() => {
    libraryRequest.current?.abort();
    const request = new AbortController();
    libraryRequest.current = request;
    setLoadingLibrary(true);
    setError(null);
    void getHumanMotionLibrary({ signal: request.signal })
      .then((response) => {
        if (request.signal.aborted) return;
        setEntries(response.entries);
      })
      .catch((reason: unknown) => {
        if (request.signal.aborted) return;
        setError(errorMessage(reason));
      })
      .finally(() => {
        if (!request.signal.aborted) setLoadingLibrary(false);
      });
  }, []);

  useEffect(() => {
    refreshLibrary();
  }, [libraryRevision, refreshLibrary]);

  useEffect(() => {
    return () => {
      libraryRequest.current?.abort();
      motionRequest.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (
      !importRequest ||
      handledImportRequest.current === importRequest.id ||
      (importRequest.target !== "motion-file" &&
        importRequest.target !== "motion-folder")
    ) {
      return;
    }
    handledImportRequest.current = importRequest.id;
    setProfile("mimic");
    if (importRequest.target === "motion-file") fileInput.current?.click();
    else folderInput.current?.click();
  }, [importRequest]);

  const visibleEntries = useMemo(() => {
    const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return entries.filter((entry) => {
      const motionCategory = entryCategory(entry);
      if (category !== "all" && motionCategory !== category) return false;
      const searchable = [
        entry.folder_label,
        entry.stem,
        entry.sequence_id,
        entry.dataset,
        motionCategory,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return tokens.every((token) => searchable.includes(token));
    });
  }, [category, entries, query]);

  const loadEntry = useCallback(
    (entry: MotionLibraryEntry) => {
      if (loadingKey) return;
      motionRequest.current?.abort();
      const request = new AbortController();
      motionRequest.current = request;
      const key = entryKey(entry);
      setLoadingKey(key);
      setError(null);
      setStatus(`${text("Loading", "正在加载")} ${motionLabel(entry)}…`);
      void loadMotionLibraryEntry(entry, {
        signal: request.signal,
        onUpdate: (job) => {
          if (!request.signal.aborted) {
            const progress = Math.round((job.progress ?? 0) * 100);
            setStatus(
              `${job.message || text("Loading motion…", "正在加载动作…")} ${progress}%`,
            );
          }
        },
      })
        .then((payload) => {
          if (request.signal.aborted) return;
          const stagePayload = toStageMotionPayload(payload);
          if (!stagePayload) {
            throw new Error(
              text(
                "The motion result has no preview data.",
                "动作结果中没有可预览的数据。",
              ),
            );
          }
          setStatus(`${text("Loaded", "已加载")} ${motionLabel(entry)}`);
          onMotionLoaded?.(stagePayload);
        })
        .catch((reason: unknown) => {
          if (request.signal.aborted) return;
          setError(errorMessage(reason));
          setStatus(null);
        })
        .finally(() => {
          if (!request.signal.aborted) setLoadingKey(null);
        });
    },
    [loadingKey, motionLabel, onMotionLoaded, text],
  );

  const importFiles = useCallback(
    (fileList: Iterable<File> | null) => {
      const files = fileList ? Array.from(fileList) : [];
      if (!files.length || loadingKey) return;
      motionRequest.current?.abort();
      const request = new AbortController();
      motionRequest.current = request;
      setLoadingKey(`upload:${files[0].name}`);
      setError(null);
      setStatus(
        text(
          `Uploading ${files.length} file${files.length === 1 ? "" : "s"}…`,
          `正在上传 ${files.length} 个文件…`,
        ),
      );
      void uploadMotion(files, {
        profile,
        libraryFolderLabel: uploadFolderLabel(files),
        signal: request.signal,
        onUpdate: (job) => {
          if (!request.signal.aborted) {
            const progress = Math.round((job.progress ?? 0) * 100);
            setStatus(
              `${job.message || text("Processing motion…", "正在处理动作…")} ${progress}%`,
            );
          }
        },
      })
        .then((payload) => {
          if (request.signal.aborted) return;
          const stagePayload = toStageMotionPayload(payload);
          if (!stagePayload) {
            throw new Error(
              text(
                "The motion result has no preview data.",
                "动作结果中没有可预览的数据。",
              ),
            );
          }
          setStatus(
            `${text("Loaded", "已加载")} ${payload.name || files[0].name}`,
          );
          onMotionLoaded?.(stagePayload);
          refreshLibrary();
        })
        .catch((reason: unknown) => {
          if (request.signal.aborted) return;
          setError(errorMessage(reason));
          setStatus(null);
        })
        .finally(() => {
          if (!request.signal.aborted) setLoadingKey(null);
        });
    },
    [loadingKey, onMotionLoaded, profile, refreshLibrary, text],
  );

  return (
    <InspectorPage title={text("Motion", "动作")}>
      <div
        className="flex shrink-0 flex-col gap-2.5"
        data-tutorial="motion-import"
      >
        <SegmentedControl
          label={text("Motion import type", "动作导入类型")}
          items={profiles}
          value={profile}
          onValueChange={setProfile}
        />

        <ImportDropzone
          label={text(`${profile} import area`, `${profile} 导入区`)}
          icon={selected.icon}
          title={text(selected.prompt, selected.promptZh)}
          disabled={Boolean(loadingKey)}
          onFiles={importFiles}
        >
          {selected.acceptsFile && (
            <Button
              size="sm"
              disabled={Boolean(loadingKey)}
              onClick={() => fileInput.current?.click()}
            >
              {text("Choose file", "选择文件")}
            </Button>
          )}
          <Button
            size="sm"
            disabled={Boolean(loadingKey)}
            onClick={() => folderInput.current?.click()}
          >
            {text("Choose folder", "选择文件夹")}
          </Button>
          <input
            ref={fileInput}
            className="hidden"
            type="file"
            accept=".bvh,.glb,.gltf,.npz,.npy,.pkl,.pt"
            onChange={(event) => {
              importFiles(event.currentTarget.files);
              event.currentTarget.value = "";
            }}
          />
          <input
            ref={folderInput}
            className="hidden"
            type="file"
            multiple
            {...({ webkitdirectory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
            onChange={(event) => {
              importFiles(event.currentTarget.files);
              event.currentTarget.value = "";
            }}
          />
        </ImportDropzone>
        <div className="flex min-h-[30px] items-center gap-2">
          <p
            className="min-w-0 flex-1 truncate text-xs text-muted-foreground"
            aria-live="polite"
            title={status || undefined}
          >
            {status || ""}
          </p>
        </div>
        <ValidationSummary
          items={motionValidationFacts(currentMotion ?? null, text)}
          label={text("Loaded motion validation", "已加载动作校验")}
        />
      </div>

      <section
        className="flex min-h-40 flex-[1_1_220px] flex-col gap-2"
        aria-labelledby="motion-library-title"
        data-tutorial="motion-library"
      >
        <div className="flex items-center justify-between gap-2">
          <h2
            id="motion-library-title"
            className="text-[19px] leading-tight font-bold tracking-normal text-foreground"
          >
            {text("Human Motion Library", "人体动作资源库")}
          </h2>
          <RefreshButton
            label={text("Refresh Motion Library", "刷新动作资源库")}
            busy={loadingLibrary}
            variant="ghost"
            onClick={refreshLibrary}
            disabled={Boolean(loadingKey)}
          />
        </div>
        <SearchField
          label={text("Search the Motion Library", "搜索动作资源库")}
          placeholder={text("Search motions...", "搜索动作……")}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          disabled={loadingLibrary}
        />
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-1.5">
          <select
            className={fieldClass}
            value={category}
            onChange={(event) => {
              const value = event.target.value;
              if (
                value === "all" ||
                value === "motion" ||
                value === "object" ||
                value === "terrain"
              ) {
                setCategory(value);
              }
            }}
            aria-label={text("Motion library category", "动作资源库类型")}
            disabled={loadingLibrary}
          >
            {categories.map((item) => (
              <option key={item.value} value={item.value}>
                {text(item.label, item.labelZh)}
              </option>
            ))}
          </select>
          {onOpenSettings && (
            <Button size="sm" variant="primary" onClick={onOpenSettings}>
              {text("Set directory", "设置目录")}
            </Button>
          )}
        </div>
        {error && (
          <p
            className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] leading-relaxed break-words text-danger"
            role="alert"
          >
            {error}
          </p>
        )}
        <div
          className="min-h-[120px] flex-[1_1_220px] overflow-y-auto rounded-md border border-border-subtle bg-surface p-1"
          aria-live="polite"
          aria-busy={loadingLibrary || Boolean(loadingKey)}
        >
          {loadingLibrary ? (
            <p className="p-2 text-xs text-muted-foreground">
              {text("Loading Motion Library…", "正在加载动作资源库…")}
            </p>
          ) : !entries.length ? (
            <p className="p-2 text-xs text-muted-foreground">
              {text("No recognizable motions are available.", "没有可识别的动作。")}
            </p>
          ) : !visibleEntries.length ? (
            <p className="p-2 text-xs text-muted-foreground">
              {text(
                `No motions match “${query}”.`,
                `没有动作匹配“${query}”。`,
              )}
            </p>
          ) : (
            <ul
              className="grid gap-0.5"
              aria-label={text("Motion Library entries", "动作资源库条目")}
            >
              {visibleEntries.slice(0, 300).map((entry) => {
                const key = entryKey(entry);
                const active = selectedKey === key;
                const busy = loadingKey === key;
                const motionCategory = entryCategory(entry);
                return (
                  <li
                    key={key}
                    className="min-w-0 list-none"
                  >
                    <button
                      type="button"
                      className="grid min-h-12 w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-md border border-transparent bg-transparent px-2 py-1.5 text-left text-foreground transition-colors hover:border-border-subtle hover:bg-background focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring data-[active=true]:border-primary data-[active=true]:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
                      data-active={active}
                      aria-current={active ? "true" : undefined}
                      aria-label={text(
                        `Load motion ${motionLabel(entry)}`,
                        `加载动作 ${motionLabel(entry)}`,
                      )}
                      disabled={Boolean(loadingKey)}
                      onClick={() => loadEntry(entry)}
                    >
                      <span
                        className={`rounded-sm px-1.5 py-1 text-[10px] font-semibold uppercase ${categoryBadgeClass[motionCategory]}`}
                      >
                        {text(
                          categories.find((item) => item.value === motionCategory)
                            ?.label ?? motionCategory,
                          categories.find((item) => item.value === motionCategory)
                            ?.labelZh ?? motionCategory,
                        )}
                      </span>
                      <span className="grid min-w-0 gap-0.5">
                        <strong className="truncate text-[13px] font-semibold">
                          {motionLabel(entry)}
                        </strong>
                        <small className="truncate text-[11px] text-muted-foreground">
                          {[entry.folder_label, entry.dataset]
                            .filter(Boolean)
                            .join(" · ") ||
                            text("Motion Library", "动作资源库")}
                        </small>
                      </span>
                      <span className="shrink-0 text-[11px] text-muted-foreground">
                        {busy
                          ? text("Loading…", "加载中…")
                          : active
                            ? text("Loaded", "已加载")
                            : text("Load", "加载")}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>
    </InspectorPage>
  );
}
