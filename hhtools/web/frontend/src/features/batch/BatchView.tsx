import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { InspectorPage } from "@/components/Inspector";
import { RefreshButton } from "@/components/RefreshButton";
import { SegmentedControl } from "@/components/SegmentedControl";
import {
  getHumanMotionLibrary,
  type MotionLibraryEntry,
} from "@/features/motion/api";
import {
  getRobotLibrary,
  type RobotSummary,
} from "@/features/robot/api";
import { useLocaleText } from "@/LocaleProvider";

import { HumanBatchView } from "./HumanBatchView";
import { RobotBatchView } from "./RobotBatchView";
import { appendUniqueEntries } from "./model";

type BatchMode = "v2m" | "h2r" | "r2r";

const modes = [
  { id: "v2m", label: "V2M" },
  { id: "h2r", label: "H2R" },
  { id: "r2r", label: "R2R" },
] as const;

const VideoBatchView = lazy(async () => ({
  default: (await import("./VideoBatchView")).VideoBatchView,
}));

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
/**
 * Batch owns one lightweight catalog snapshot and three independent drafts.
 * Hidden modes remain mounted so switching workflows never destroys a queue or job.
 */
export function BatchView({
  active = true,
  runtimeRevision = 0,
  humanEntries,
  onHumanEntriesChange,
  onMotionLibraryChange,
}: {
  active?: boolean;
  runtimeRevision?: number;
  humanEntries: readonly MotionLibraryEntry[];
  onHumanEntriesChange(entries: readonly MotionLibraryEntry[]): void;
  onMotionLibraryChange?: () => void;
}) {
  const text = useLocaleText();
  const [mode, setMode] = useState<BatchMode>("h2r");
  const [videoModeMounted, setVideoModeMounted] = useState(false);
  const [motions, setMotions] = useState<readonly MotionLibraryEntry[]>([]);
  const [robots, setRobots] = useState<readonly RobotSummary[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogBusy, setCatalogBusy] = useState(false);
  const catalogRequest = useRef<AbortController | null>(null);
  const catalogPrefetched = useRef(false);
  const humanEntriesRef = useRef(humanEntries);
  humanEntriesRef.current = humanEntries;

  const refreshCatalogs = useCallback(() => {
    catalogRequest.current?.abort();
    const request = new AbortController();
    catalogRequest.current = request;
    setCatalogBusy(true);
    setCatalogError(null);
    void Promise.all([
      getHumanMotionLibrary({ signal: request.signal }),
      getRobotLibrary({ signal: request.signal }),
    ])
      .then(([motionLibrary, robotLibrary]) => {
        if (request.signal.aborted) return;
        setMotions(motionLibrary.entries);
        setRobots(robotLibrary.robots);
      })
      .catch((reason: unknown) => {
        if (!request.signal.aborted) setCatalogError(errorMessage(reason));
      })
      .finally(() => {
        if (!request.signal.aborted) {
          catalogPrefetched.current = true;
          setCatalogBusy(false);
        }
      });
  }, []);

  useEffect(() => {
    if (!active && catalogPrefetched.current) {
      catalogRequest.current?.abort();
      return;
    }
    refreshCatalogs();
  }, [active, refreshCatalogs]);

  useEffect(() => () => catalogRequest.current?.abort(), []);

  useEffect(() => {
    if (mode === "v2m") setVideoModeMounted(true);
  }, [mode]);

  function addPublishedMotion(entry: MotionLibraryEntry): void {
    const next = appendUniqueEntries(humanEntriesRef.current, [entry]);
    humanEntriesRef.current = next;
    onHumanEntriesChange(next);
    onMotionLibraryChange?.();
    refreshCatalogs();
  }

  return (
    <InspectorPage
      title={text("Batch", "批处理")}
      headerAction={
        <RefreshButton
          label={text("Refresh Batch catalogs", "刷新批处理资源库")}
          busy={catalogBusy}
          variant="ghost"
          onClick={refreshCatalogs}
        />
      }
    >
      <SegmentedControl
        label={text("Batch workflow", "批处理流程")}
        items={modes}
        value={mode}
        onValueChange={setMode}
      />

      {(videoModeMounted || mode === "v2m") && (
        <div hidden={mode !== "v2m"}>
          <Suspense
            fallback={
              <div
                className="flex min-h-24 items-center justify-center gap-2 text-xs text-muted-foreground"
                role="status"
              >
                <span
                  className="size-4 animate-spin bg-current [mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat]"
                  aria-hidden="true"
                />
                <span>{text("Loading Video Batch…", "正在加载视频批处理……")}</span>
              </div>
            }
          >
            <VideoBatchView
              onMotionPublished={addPublishedMotion}
              runtimeRevision={runtimeRevision}
            />
          </Suspense>
        </div>
      )}
      <div hidden={mode !== "h2r"}>
        <HumanBatchView
          active={active && mode === "h2r"}
          library={motions}
          robots={robots}
          entries={humanEntries}
          onEntriesChange={onHumanEntriesChange}
          catalogError={catalogError}
        />
      </div>
      <div hidden={mode !== "r2r"}>
        <RobotBatchView
          active={active && mode === "r2r"}
          robots={robots}
          catalogError={catalogError}
        />
      </div>
    </InspectorPage>
  );
}
