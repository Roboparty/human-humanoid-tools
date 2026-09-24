import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ImportDropzone } from "@/components/ImportDropzone";
import { InspectorPage } from "@/components/Inspector";
import { RefreshButton } from "@/components/RefreshButton";
import { SearchField } from "@/components/SearchField";
import { ValidationSummary } from "@/components/ValidationSummary";
import { robotValidationFacts } from "@/components/validationFacts";
import { Button } from "@/components/ui/button";
import type { ApplicationImportRequest } from "@/importIntent";
import { useLocaleText } from "@/LocaleProvider";
import type { UploadFile } from "@/lib/api";

import {
  deleteRobot,
  getRobotLibrary,
  loadRobot,
  uploadRobot,
  type RobotPayload,
  type RobotSummary,
} from "./api";

/** Keep curated models in the same visual order as the previous inspector. */
const CURATED_ORDER: Readonly<Record<string, number>> = {
  g1_29dof: 0,
  roboto_origin: 1,
  agibot_x2_ultra: 2,
  asimov_1: 3,
  fourier_gr2: 4,
  berkeley_humanoid_lite: 5,
};

const CURATED_LABELS: Readonly<Record<string, string>> = {
  g1_29dof: "Unitree G1",
  roboto_origin: "ROBOTO_ORIGIN (RPO)",
  agibot_x2_ultra: "AgiBot X2",
  asimov_1: "Asimov 1",
  fourier_gr2: "Fourier GR-2",
  berkeley_humanoid_lite: "Berkeley Humanoid Lite",
};

const CURATED_ICONS: Readonly<Record<string, string>> = {
  g1_29dof: "/robot-icons/unitree-g1.webp",
  roboto_origin: "/robot-icons/roboto-origin.webp",
  agibot_x2_ultra: "/robot-icons/agibot-x2.webp",
  asimov_1: "/robot-icons/asimov-1.webp",
  fourier_gr2: "/robot-icons/fourier-gr2.webp",
  berkeley_humanoid_lite: "/robot-icons/berkeley-humanoid-lite.webp",
};

const FALLBACK_ICON = "/hhtools-robot.svg";
const CURATED_NAMES = Object.keys(CURATED_LABELS);

function mapValue<T>(map: Readonly<Record<string, T>>, key: string): T | undefined {
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined;
}

function robotLabel(robot: RobotSummary): string {
  return mapValue(CURATED_LABELS, robot.name) ?? (robot.display_name || robot.name);
}

function sortRobots(left: RobotSummary, right: RobotSummary): number {
  const leftOrder = mapValue(CURATED_ORDER, left.name);
  const rightOrder = mapValue(CURATED_ORDER, right.name);
  if (leftOrder !== undefined || rightOrder !== undefined) {
    if (leftOrder === undefined) return 1;
    if (rightOrder === undefined) return -1;
    return leftOrder - rightOrder;
  }
  return robotLabel(left).localeCompare(robotLabel(right));
}

function robotIcon(name: string): string {
  return mapValue(CURATED_ICONS, name) ?? FALLBACK_ICON;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function uploadName(file: File): string {
  return file.name
    .replace(/\.urdf$/i, "")
    .replace(/[^a-z0-9_]/gi, "_")
    .toLowerCase() || "uploaded_robot";
}

export function RobotView({
  currentRobot,
  onRobotLoaded,
  importRequest,
}: {
  /** App-owned stable input; failed replacements leave it untouched. */
  currentRobot?: RobotPayload | null;
  /** App uses this to publish the server payload to the shared Stage. */
  onRobotLoaded?: (payload: RobotPayload | null) => void;
  /** App-owned File-menu intent; this mounted view owns its input elements. */
  importRequest?: ApplicationImportRequest | null;
}) {
  const text = useLocaleText();
  const [robots, setRobots] = useState<RobotSummary[]>([]);
  const [search, setSearch] = useState("");
  const [loadingName, setLoadingName] = useState<string | null>(null);
  const [loadingLibrary, setLoadingLibrary] = useState(true);
  const [importing, setImporting] = useState(false);
  const [urdf, setUrdf] = useState<UploadFile | null>(null);
  const [meshes, setMeshes] = useState<readonly UploadFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const libraryRequest = useRef<AbortController | null>(null);
  const robotRequest = useRef<AbortController | null>(null);
  const urdfInput = useRef<HTMLInputElement | null>(null);
  const meshInput = useRef<HTMLInputElement | null>(null);
  const handledImportRequest = useRef<number | null>(null);

  const refresh = useCallback(() => {
    libraryRequest.current?.abort();
    const request = new AbortController();
    libraryRequest.current = request;
    setLoadingLibrary(true);
    setError(null);
    void getRobotLibrary({ signal: request.signal })
      .then((response) => {
        if (request.signal.aborted) return;
        setRobots([...response.robots]);
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
    refresh();
    return () => {
      libraryRequest.current?.abort();
      robotRequest.current?.abort();
    };
  }, [refresh]);

  useEffect(() => {
    if (
      !importRequest ||
      handledImportRequest.current === importRequest.id ||
      (importRequest.target !== "robot-urdf" &&
        importRequest.target !== "robot-mesh-folder")
    ) {
      return;
    }
    handledImportRequest.current = importRequest.id;
    if (importRequest.target === "robot-urdf") urdfInput.current?.click();
    else meshInput.current?.click();
  }, [importRequest]);

  const filteredRobots = useMemo(() => {
    const tokens = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return [...robots]
      .sort(sortRobots)
      .filter((robot) => {
        const haystack = [
          robot.name,
          robotLabel(robot),
          robot.display_name,
          String(robot.num_dof),
          robot.builtin ? "built-in builtin 内置" : "imported custom 导入",
        ]
          .join(" ")
          .toLowerCase();
        return tokens.every((token) => haystack.includes(token));
      });
  }, [robots, search]);

  const loadedName = currentRobot?.name ?? null;
  const loadedRobot = loadedName
    ? robots.find((robot) => robot.name === loadedName)
    : undefined;
  const busy = importing || Boolean(loadingName);

  const finishImport = useCallback(
    async (selectedUrdf: UploadFile, selectedMeshes: readonly UploadFile[]) => {
      robotRequest.current?.abort();
      const request = new AbortController();
      robotRequest.current = request;
      setImporting(true);
      setError(null);
      try {
        const payload = await uploadRobot(
          [selectedUrdf, ...selectedMeshes],
          uploadName(selectedUrdf),
          { signal: request.signal },
        );
        if (request.signal.aborted) return;
        setUrdf(null);
        setMeshes([]);
        onRobotLoaded?.(payload);
        refresh();
      } catch (reason) {
        if (!request.signal.aborted) setError(errorMessage(reason));
      } finally {
        if (!request.signal.aborted) setImporting(false);
      }
    },
    [onRobotLoaded, refresh],
  );

  // Either picker may run first; upload starts only after both halves exist.
  const receiveUrdf = useCallback(
    (files: readonly UploadFile[]) => {
      if (busy) return;
      const selectedUrdf = files.find((file) =>
        file.name.toLowerCase().endsWith(".urdf"),
      );
      if (!selectedUrdf) {
        setError(
          text("No .urdf file was found.", "没有找到 .urdf 文件。"),
        );
        return;
      }
      const sidecars = files.filter(
        (file) =>
          file !== selectedUrdf && !file.name.toLowerCase().endsWith(".urdf"),
      );
      const selectedMeshes = sidecars.length ? sidecars : meshes;
      setUrdf(selectedUrdf);
      if (sidecars.length) setMeshes(sidecars);
      setError(null);
      if (selectedMeshes.length) void finishImport(selectedUrdf, selectedMeshes);
    },
    [busy, finishImport, meshes, text],
  );

  const receiveMeshes = useCallback(
    (files: readonly UploadFile[]) => {
      if (busy) return;
      const sidecars = files.filter(
        (file) => !file.name.toLowerCase().endsWith(".urdf"),
      );
      if (!sidecars.length) {
        setError(text("No mesh assets were found.", "没有找到网格资源。"));
        return;
      }
      setMeshes(sidecars);
      setError(null);
      if (urdf) void finishImport(urdf, sidecars);
    },
    [busy, finishImport, text, urdf],
  );

  const remove = useCallback(
    (robot: RobotSummary) => {
      if (!robot.deletable || busy) return;
      if (!window.confirm(
        text(
          `Remove “${robotLabel(robot)}” from the Robot Library?\nThis permanently deletes its local folder.`,
          `要从机器人资源库移除“${robotLabel(robot)}”吗？\n这将永久删除其本地文件夹。`,
        ),
      )) return;
      robotRequest.current?.abort();
      const request = new AbortController();
      robotRequest.current = request;
      setLoadingName(robot.name);
      setError(null);
      void deleteRobot(robot.name, { signal: request.signal })
        .then(() => {
          if (request.signal.aborted) return;
          if (currentRobot?.name === robot.name) onRobotLoaded?.(null);
          refresh();
        })
        .catch((reason: unknown) => {
          if (!request.signal.aborted) setError(errorMessage(reason));
        })
        .finally(() => {
          if (!request.signal.aborted) setLoadingName(null);
        });
    },
    [busy, currentRobot?.name, onRobotLoaded, refresh, text],
  );

  const load = useCallback(
    (robot: RobotSummary) => {
      if (!robot.has_urdf || loadingName) return;
      robotRequest.current?.abort();
      const request = new AbortController();
      robotRequest.current = request;
      setLoadingName(robot.name);
      setError(null);
      void loadRobot(robot.name, { signal: request.signal })
        .then((payload) => {
          if (request.signal.aborted) return;
          onRobotLoaded?.(payload);
        })
        .catch((reason: unknown) => {
          if (request.signal.aborted) return;
          setError(errorMessage(reason));
        })
        .finally(() => {
          if (!request.signal.aborted) setLoadingName(null);
        });
    },
    [loadingName, onRobotLoaded],
  );

  return (
    <InspectorPage title={text("Robot", "机器人")}>
      <div
        className="flex shrink-0 flex-col gap-2.5"
        data-tutorial="robot-import"
      >
        <ImportDropzone
          label={text("URDF import area", "URDF 导入区")}
          icon="/icons/robot/file.svg"
          title={text("1 · URDF file", "1 · URDF 文件")}
          className="min-h-[120px] px-9 py-3.5"
          disabled={busy}
          onFiles={receiveUrdf}
        >
          <Button size="sm" disabled={busy} onClick={() => urdfInput.current?.click()}>
            {text("Choose .urdf", "选择 .urdf")}
          </Button>
          <input
            ref={urdfInput}
            className="hidden"
            type="file"
            accept=".urdf"
            onChange={(event) => {
              receiveUrdf(Array.from(event.currentTarget.files ?? []) as UploadFile[]);
              event.currentTarget.value = "";
            }}
          />
        </ImportDropzone>
        <ImportDropzone
          label={text("Robot mesh import area", "机器人网格导入区")}
          icon="/icons/robot/folder.svg"
          title={text("2 · Mesh folder", "2 · 网格文件夹")}
          className="min-h-[120px] px-9 py-3.5"
          disabled={busy}
          onFiles={receiveMeshes}
        >
          <Button
            size="sm"
            disabled={busy}
            onClick={() => meshInput.current?.click()}
          >
            {text("Choose mesh folder", "选择网格文件夹")}
          </Button>
          <input
            ref={meshInput}
            className="hidden"
            type="file"
            multiple
            disabled={busy}
            onChange={(event) => {
              receiveMeshes(Array.from(event.currentTarget.files ?? []) as UploadFile[]);
              event.currentTarget.value = "";
            }}
            {...{ webkitdirectory: "" }}
          />
        </ImportDropzone>
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {importing
            ? text(
                `Importing ${urdf?.name || "robot"}...`,
                `正在导入 ${urdf?.name || "机器人"}……`,
              )
            : urdf
              ? `URDF: ${urdf.name} · ${
                  meshes.length
                    ? text(
                        `${meshes.length} assets`,
                        `${meshes.length} 个资源`,
                      )
                    : text("choose the mesh folder", "请选择网格文件夹")
                }`
              : meshes.length
                ? text(
                    `${meshes.length} mesh asset${meshes.length === 1 ? "" : "s"} · choose the .urdf file`,
                    `${meshes.length} 个网格资源 · 请选择 .urdf 文件`,
                  )
              : loadedRobot
                ? text(
                    `Loaded: ${robotLabel(loadedRobot)}`,
                    `已加载：${robotLabel(loadedRobot)}`,
                  )
                : currentRobot
                  ? text(
                      `Loaded: ${currentRobot.display_name}`,
                      `已加载：${currentRobot.display_name}`,
                    )
                  : text("No URDF selected.", "尚未选择 URDF。")}
        </p>
        <ValidationSummary
          items={robotValidationFacts(currentRobot ?? null, text)}
          label={text("Loaded robot validation", "已加载机器人校验")}
        />
      </div>

      <section
        className="flex min-h-[220px] flex-1 flex-col gap-2"
        aria-labelledby="robot-library-title"
      >
        <div className="flex items-center justify-between gap-2">
          <h2
            id="robot-library-title"
            className="text-[19px] leading-tight font-bold tracking-normal text-foreground"
          >
            {text("Robot Library", "机器人资源库")}
          </h2>
          <RefreshButton
            label={text("Refresh Robot Library", "刷新机器人资源库")}
            busy={loadingLibrary}
            variant="ghost"
            onClick={refresh}
            disabled={Boolean(loadingName)}
          />
        </div>
        <SearchField
          label={text("Search the Robot Library", "搜索机器人资源库")}
          placeholder={text("Search robots...", "搜索机器人……")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          disabled={loadingLibrary}
        />
        {error && (
          <p className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] leading-relaxed text-danger break-words" role="alert">
            {error}
          </p>
        )}
        <div
          className="min-h-[138px] flex-1 overflow-y-auto rounded-md border border-border-subtle bg-surface p-1"
          aria-live="polite"
          aria-busy={loadingLibrary || Boolean(loadingName)}
        >
          {loadingLibrary ? (
            <p className="p-2 text-xs text-muted-foreground">
              {text("Loading robots...", "正在加载机器人……")}
            </p>
          ) : !robots.length ? (
            <div className="grid gap-1 p-2 text-xs text-muted-foreground">
              <p>
                {text(
                  "No robot models are available.",
                  "没有可用的机器人模型。",
                )}
              </p>
              <p className="break-words text-[11px] leading-relaxed">
                {text(
                  "Import a URDF and its mesh folder, then refresh.",
                  "导入 URDF 及其网格文件夹，然后刷新。",
                )}
              </p>
              <p className="break-words text-[11px] leading-relaxed">
                {text("Curated presets: ", "内置模型：")}{CURATED_NAMES.map(
                  (name) => CURATED_LABELS[name],
                ).join(", ")}.
              </p>
            </div>
          ) : !filteredRobots.length ? (
            <p className="p-2 text-xs text-muted-foreground">
              {text(
                `No robots match “${search}”.`,
                `没有机器人匹配“${search}”。`,
              )}
            </p>
          ) : (
            <ul
              className="grid gap-0.5"
              aria-label={text("Robot models", "机器人模型")}
            >
              {filteredRobots.map((robot) => {
                const active = loadedName === robot.name;
                const busy = loadingName === robot.name;
                const unavailable = !robot.has_urdf;
                return (
                  <li
                    key={robot.name}
                    className="grid min-w-0 list-none grid-cols-[minmax(0,1fr)_auto] items-center"
                  >
                    <button
                      type="button"
                      className="grid min-h-12 w-full grid-cols-[32px_minmax(0,1fr)_auto] items-center gap-2 rounded-md border border-transparent bg-transparent px-2 py-1.5 text-left text-foreground transition-colors hover:border-border-subtle hover:bg-background focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring data-[active=true]:border-primary data-[active=true]:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                      data-active={active}
                      aria-current={active ? "true" : undefined}
                      aria-label={text(
                        `Load robot ${robotLabel(robot)}`,
                        `加载机器人 ${robotLabel(robot)}`,
                      )}
                      disabled={unavailable || busy}
                      onClick={() => load(robot)}
                    >
                      <img
                        className="size-8 rounded-sm object-cover"
                        src={robotIcon(robot.name)}
                        alt=""
                        aria-hidden="true"
                        onError={(event) => {
                          const image = event.currentTarget;
                          if (image.src.endsWith(FALLBACK_ICON)) return;
                          image.src = FALLBACK_ICON;
                        }}
                      />
                      <span className="grid min-w-0 gap-0.5">
                        <strong className="truncate text-[13px] font-semibold">
                          {robotLabel(robot)}
                        </strong>
                        <small className="truncate text-[11px] text-muted-foreground">
                          {robot.num_dof} DoF · {robot.builtin
                            ? text("Built-in", "内置")
                            : text("Imported", "已导入")}
                          {unavailable
                            ? text(" · URDF missing", " · 缺少 URDF")
                            : ""}
                        </small>
                      </span>
                      {(busy || unavailable) && (
                        <span className="shrink-0 text-[11px] text-muted-foreground">
                          {busy
                            ? text("Loading...", "加载中……")
                            : text("Unavailable", "不可用")}
                        </span>
                      )}
                    </button>
                    {robot.deletable && !robot.builtin && (
                      <button
                        type="button"
                        className="size-8 rounded-md text-lg leading-none text-muted-foreground hover:bg-danger-muted hover:text-danger focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
                        title={text(
                          "Remove from Robot Library",
                          "从机器人资源库移除",
                        )}
                        aria-label={text(
                          `Delete robot ${robotLabel(robot)}`,
                          `删除机器人 ${robotLabel(robot)}`,
                        )}
                        disabled={busy}
                        onClick={() => remove(robot)}
                      >
                        ×
                      </button>
                    )}
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
