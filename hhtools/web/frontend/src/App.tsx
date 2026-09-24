import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";

import {
  storeTheme,
  storedThemeOverride,
  viewForImport,
  type ApplicationImportRequest,
  type ApplicationImportTarget,
  type ApplicationTheme,
} from "./appCommands";
import {
  ApplicationDialogs,
  type ApplicationDialog,
} from "./components/ApplicationDialogs";
import { Inspector } from "./components/Inspector";
import { Navbar } from "./components/Navbar";
import { Sidebar } from "./components/Sidebar";
import { TutorialOverlay } from "./components/TutorialOverlay";
import { LocaleProvider, useLocaleText } from "./LocaleProvider";
import {
  storeLocale,
  storedLocaleOverride,
  systemLocale,
  type WorkspaceLocale,
} from "./localization";
import { AnalysisView } from "./features/analysis/AnalysisView";
import type { AnalysisRobotPreview } from "./features/analysis/api";
import {
  storedForceReanalysis,
  storeForceReanalysis,
} from "./features/analysis/preferences";
import { BatchView } from "./features/batch/BatchView";
import { HumanToRobotView } from "./features/h2r/HumanToRobotView";
import { MotionView } from "./features/motion/MotionView";
import {
  invalidateMotionLibrary,
  type MotionLibraryEntry,
} from "./features/motion/api";
import { RobotToRobotView } from "./features/r2r/RobotToRobotView";
import { RobotView } from "./features/robot/RobotView";
import {
  rememberTutorialSeen,
  shouldAutoOpenTutorial,
  type TutorialPersistenceBridge,
  type TutorialStep,
} from "./features/tutorial/model";
import {
  retargetExportUrl,
  type CalibrationPose,
  type RetargetResult as H2rResult,
  type ScaledPreviewResult as H2rScaledPreview,
} from "./features/h2r/api";
import {
  r2rExportUrl,
  type R2rRetargetResult,
  type R2rScenePayload,
  type R2rSourceResult,
} from "./features/r2r/api";
import {
  comparisonLayers,
  storedComparisonPreset,
  storeComparisonPreset,
  type ComparisonPreset,
} from "./features/result/comparison";
import { TaskDrawer } from "./features/tasks/TaskDrawer";
import type { ViewId } from "./navigation";
import { Stage } from "./stage/Stage";
import { DEFAULT_CALIBRATION_DISPLAY } from "./stage/calibrationDisplay";
import type { CalibrationInteractionModel } from "./stage/calibrationInteraction";
import type { StagePresentation } from "./stage/presentation";
import type {
  StageMotionPayload,
  StageR2rPresentationPayload,
  StageRobotPayload,
  StageRobotTrajectoryPayload,
} from "./stage/types";
import {
  storedWorkspaceLayout,
  storeWorkspaceLayout,
} from "./workspaceLayout";
import { preloadCoreWorkspace } from "./workspaceBootstrap";

const VideoToMotionView = lazy(async () => ({
  default: (await import("./features/video-to-motion/VideoToMotionView"))
    .VideoToMotionView,
}));

function motionWithScene(
  motion: StageMotionPayload | null | undefined,
  scene: R2rScenePayload | H2rResult["scaled_scene"],
  meshSource?: StageMotionPayload["object_mesh_source"],
): StageMotionPayload | null {
  if (!motion && !scene) return null;
  return {
    ...motion,
    positions: motion?.positions ?? [],
    parent_indices: motion?.parent_indices ?? [],
    terrain: scene?.terrain ?? motion?.terrain,
    objects: scene?.objects ?? motion?.objects,
    object_mesh_source: meshSource,
  };
}

function calibrationTrajectory(
  pose: CalibrationPose,
  robot: StageRobotPayload | null,
): StageRobotTrajectoryPayload {
  return {
    frames: [
      {
        links: pose.link_transforms,
        mesh_z_lift: pose.ground_offset_z - (robot?.ground_offset_z ?? 0),
      },
    ],
  };
}

interface ApplicationDesktopBridge extends Partial<TutorialPersistenceBridge> {
  readonly exitApplication?: () => Promise<void>;
  readonly selectDirectory?: () => Promise<string | null>;
}

function desktopBridge(): ApplicationDesktopBridge | undefined {
  return (
    window as Window & { readonly hhtoolsDesktop?: ApplicationDesktopBridge }
  ).hhtoolsDesktop;
}

function preferredSystemTheme(): ApplicationTheme {
  return typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function preferredSystemLocale(): WorkspaceLocale {
  return systemLocale([
    ...window.navigator.languages,
    window.navigator.language,
  ]);
}

function VideoWorkspaceLoading() {
  const text = useLocaleText();
  return (
    <div
      className="flex h-full items-center justify-center gap-2 text-xs text-muted-foreground"
      role="status"
    >
      <span
        className="size-4 animate-spin bg-current [mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat]"
        aria-hidden="true"
      />
      <span>{text("Loading Video to Motion…", "正在加载视频转动作……")}</span>
    </div>
  );
}

export function App() {
  const themeOverride = useRef<ApplicationTheme | null | undefined>(undefined);
  const localeOverride = useRef<WorkspaceLocale | null | undefined>(undefined);
  if (themeOverride.current === undefined) {
    themeOverride.current = storedThemeOverride(window.localStorage);
  }
  if (localeOverride.current === undefined) {
    localeOverride.current = storedLocaleOverride(window.localStorage);
  }

  const [activeView, setActiveView] = useState<ViewId>("motion");
  const [coreWorkspaceReady, setCoreWorkspaceReady] = useState(false);
  const [videoToMotionMounted, setVideoToMotionMounted] = useState(false);
  const [theme, setTheme] = useState<ApplicationTheme>(() =>
    themeOverride.current ?? preferredSystemTheme(),
  );
  const [locale, setLocale] = useState<WorkspaceLocale>(() =>
    localeOverride.current ?? preferredSystemLocale(),
  );
  const [layout, setLayout] = useState(() =>
    storedWorkspaceLayout(window.localStorage),
  );
  const [forceAnalysis, setForceAnalysis] = useState(() =>
    storedForceReanalysis(window.localStorage),
  );
  const [dialog, setDialog] = useState<ApplicationDialog>(null);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [tutorialStep, setTutorialStep] = useState<TutorialStep["id"] | null>(
    null,
  );
  const tutorialCheck = useRef<Promise<boolean> | null>(null);
  const tutorialAutoTimer = useRef(0);
  const tutorialHandled = useRef(false);
  const [importRequest, setImportRequest] =
    useState<ApplicationImportRequest | null>(null);
  const nextImportRequestId = useRef(0);
  const exportLink = useRef<HTMLAnchorElement>(null);
  const [motionLibraryRevision, setMotionLibraryRevision] = useState(0);
  const [gvhmrRevision, setGvhmrRevision] = useState(0);
  const [workspaceMotion, setWorkspaceMotion] =
    useState<StageMotionPayload | null>(null);
  const [workspaceRobot, setWorkspaceRobot] =
    useState<StageRobotPayload | null>(null);
  const [humanBatchEntries, setHumanBatchEntries] = useState<
    readonly MotionLibraryEntry[]
  >([]);
  const [analysisMotion, setAnalysisMotion] =
    useState<StageMotionPayload | null>(null);
  const [analysisRobotPreview, setAnalysisRobotPreview] =
    useState<AnalysisRobotPreview | null>(null);
  const [stageMotion, setStageMotion] = useState<StageMotionPayload | null>(null);
  const [stageRobot, setStageRobot] = useState<StageRobotPayload | null>(null);
  const [stageRobotTrajectory, setStageRobotTrajectory] =
    useState<StageRobotTrajectoryPayload | null>(null);
  const [stageScaledMotion, setStageScaledMotion] =
    useState<StageMotionPayload | null>(null);
  const [h2rResult, setH2rResult] = useState<H2rResult | null>(null);
  const [h2rPreview, setH2rPreview] = useState<H2rScaledPreview | null>(null);
  const [h2rCalibrationReference, setH2rCalibrationReference] =
    useState<StageMotionPayload | null>(null);
  const [h2rCalibrationPose, setH2rCalibrationPose] =
    useState<CalibrationPose | null>(null);
  const [h2rCalibrationDisplay, setH2rCalibrationDisplay] = useState(
    DEFAULT_CALIBRATION_DISPLAY,
  );
  const [h2rCalibrationInteraction, setH2rCalibrationInteraction] =
    useState<CalibrationInteractionModel | null>(null);
  const [h2rComparisonPreset, setH2rComparisonPreset] =
    useState<ComparisonPreset>(() =>
      storedComparisonPreset(window.localStorage, "h2r"),
    );
  const [r2rSourceRobot, setR2rSourceRobot] =
    useState<StageRobotPayload | null>(null);
  const [r2rTargetRobot, setR2rTargetRobot] =
    useState<StageRobotPayload | null>(null);
  const [r2rSourceResult, setR2rSourceResult] =
    useState<R2rSourceResult | null>(null);
  const [r2rResult, setR2rResult] =
    useState<R2rRetargetResult | null>(null);
  const [r2rCalibrationPose, setR2rCalibrationPose] =
    useState<CalibrationPose | null>(null);
  const [r2rCalibrationReference, setR2rCalibrationReference] =
    useState<StageMotionPayload | null>(null);
  const [r2rCalibrationDisplay, setR2rCalibrationDisplay] = useState(
    DEFAULT_CALIBRATION_DISPLAY,
  );
  const [r2rCalibrationInteraction, setR2rCalibrationInteraction] =
    useState<CalibrationInteractionModel | null>(null);
  const [r2rComparisonPreset, setR2rComparisonPreset] =
    useState<ComparisonPreset>(() =>
      storedComparisonPreset(window.localStorage, "r2r"),
    );
  const noteMotionLibraryChanged = useCallback(() => {
    invalidateMotionLibrary();
    setMotionLibraryRevision((revision) => revision + 1);
  }, []);

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const request = new AbortController();
    let mounted = true;
    const timeout = window.setTimeout(() => request.abort(), 5_000);
    void preloadCoreWorkspace(request.signal).then(() => {
      window.clearTimeout(timeout);
      if (mounted) setCoreWorkspaceReady(true);
    });
    return () => {
      mounted = false;
      window.clearTimeout(timeout);
      request.abort();
    };
  }, []);

  useEffect(() => {
    if (activeView === "video-to-motion") setVideoToMotionMounted(true);
  }, [activeView]);

  useLayoutEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const preference = window.matchMedia("(prefers-color-scheme: dark)");
    const syncSystemTheme = (event: MediaQueryListEvent) => {
      if (themeOverride.current === null) {
        setTheme(event.matches ? "dark" : "light");
      }
    };
    preference.addEventListener("change", syncSystemTheme);
    return () => preference.removeEventListener("change", syncSystemTheme);
  }, []);

  useEffect(() => {
    const syncSystemLocale = () => {
      if (localeOverride.current === null) {
        setLocale(preferredSystemLocale());
      }
    };
    window.addEventListener("languagechange", syncSystemLocale);
    return () => window.removeEventListener("languagechange", syncSystemLocale);
  }, []);

  const toggleTheme = useCallback(() => {
    const next = theme === "light" ? "dark" : "light";
    themeOverride.current = next;
    setTheme(next);
    storeTheme(window.localStorage, next);
  }, [theme]);

  const changeLocale = useCallback((next: WorkspaceLocale) => {
    localeOverride.current = next;
    setLocale(next);
    storeLocale(window.localStorage, next);
  }, []);

  useEffect(() => {
    storeWorkspaceLayout(window.localStorage, layout);
  }, [layout]);

  useEffect(() => {
    storeForceReanalysis(window.localStorage, forceAnalysis);
  }, [forceAnalysis]);

  useEffect(() => {
    tutorialCheck.current ??= shouldAutoOpenTutorial(
      window.localStorage,
      desktopBridge(),
    );
    let active = true;
    void tutorialCheck.current.then(async (shouldOpen) => {
      if (!active) return;
      await rememberTutorialSeen(window.localStorage, desktopBridge());
      if (!active || !shouldOpen || tutorialHandled.current) return;
      tutorialAutoTimer.current = window.setTimeout(
        () => {
          if (!tutorialHandled.current) setTutorialOpen(true);
        },
        400,
      );
    });
    return () => {
      active = false;
      window.clearTimeout(tutorialAutoTimer.current);
    };
  }, []);

  const requestImport = useCallback((target: ApplicationImportTarget) => {
    setActiveView(viewForImport(target));
    nextImportRequestId.current += 1;
    setImportRequest({ id: nextImportRequestId.current, target });
  }, []);

  const currentExportUrl =
    activeView === "h2r" && h2rResult
      ? retargetExportUrl(h2rResult.export_token, {
          format: "csv",
          csvHeader: true,
        })
      : activeView === "r2r" && r2rResult
        ? r2rExportUrl(r2rResult.export_token, {
            format: "csv",
            csvHeader: true,
          })
        : null;

  const openTutorial = useCallback(() => {
    tutorialHandled.current = true;
    window.clearTimeout(tutorialAutoTimer.current);
    setDialog(null);
    setTutorialOpen(true);
    void rememberTutorialSeen(window.localStorage, desktopBridge());
  }, []);
  const closeTutorial = useCallback(() => {
    tutorialHandled.current = true;
    window.clearTimeout(tutorialAutoTimer.current);
    setTutorialStep(null);
    setTutorialOpen(false);
  }, []);
  const changeTutorialStep = useCallback(
    (step: TutorialStep) => setTutorialStep(step.id),
    [],
  );

  const changeComparisonPreset = useCallback(
    (workflow: "h2r" | "r2r", preset: ComparisonPreset) => {
      if (workflow === "h2r") setH2rComparisonPreset(preset);
      else setR2rComparisonPreset(preset);
      storeComparisonPreset(window.localStorage, workflow, preset);
    },
    [],
  );

  const stagePresentation: StagePresentation =
    activeView === "robot-assets"
      ? "robot"
      : activeView === "dataset-viz"
        ? "analysis"
        : activeView === "batch"
          ? "empty"
          : activeView === "h2r"
            ? h2rCalibrationReference
              ? "h2r-calibration"
              : h2rResult
                ? "h2r-result"
                : "h2r"
            : activeView === "r2r"
              ? r2rCalibrationReference
                ? "r2r-calibration"
                : r2rResult
                  ? "r2r-result"
                  : "r2r"
              : activeView;
  const r2rStage = useMemo<StageR2rPresentationPayload | null>(() => {
    if (activeView !== "r2r") return null;
    const meshSource = r2rSourceResult
      ? { kind: "r2r" as const, token: r2rSourceResult.token }
      : undefined;
    return {
      phase: r2rCalibrationReference
        ? "calibration"
        : r2rResult
          ? "result"
          : "source",
      source: {
        robot: r2rSourceRobot,
        trajectory: r2rSourceResult?.trajectory ?? null,
        skeleton: r2rSourceResult?.skeleton_preview ?? null,
        environment: motionWithScene(
          null,
          r2rSourceResult?.scaled_scene,
          meshSource,
        ),
      },
      target: {
        robot: r2rTargetRobot,
        trajectory: r2rCalibrationPose
          ? calibrationTrajectory(r2rCalibrationPose, r2rTargetRobot)
          : r2rResult?.trajectory ?? null,
        skeleton: r2rResult?.scaled_preview ?? null,
        environment: motionWithScene(
          null,
          r2rResult?.scaled_scene,
          meshSource,
        ),
      },
      calibrationReference: r2rCalibrationReference,
      sourceToken: r2rSourceResult?.token ?? null,
      resultToken: r2rResult?.export_token ?? null,
    };
  }, [
    activeView,
    r2rCalibrationPose,
    r2rCalibrationReference,
    r2rResult,
    r2rSourceResult,
    r2rSourceRobot,
    r2rTargetRobot,
  ]);

  const publishMotion = useCallback((motion: StageMotionPayload | null) => {
    setWorkspaceMotion(motion);
    setH2rResult(null);
    setH2rPreview(null);
    setH2rCalibrationReference(null);
    setH2rCalibrationPose(null);
  }, []);
  const publishRobot = useCallback((robot: StageRobotPayload | null) => {
    setWorkspaceRobot(robot);
    setH2rResult(null);
    setH2rPreview(null);
    setH2rCalibrationReference(null);
    setH2rCalibrationPose(null);
  }, []);
  const publishR2rSourceRobot = useCallback((robot: StageRobotPayload | null) => {
    setR2rSourceRobot(robot);
    setR2rCalibrationReference(null);
    if (!robot) {
      setR2rSourceResult(null);
      setR2rResult(null);
    }
  }, []);
  const publishR2rTargetRobot = useCallback((robot: StageRobotPayload | null) => {
    setR2rTargetRobot(robot);
    setR2rCalibrationPose(null);
    setR2rCalibrationReference(null);
    setR2rResult(null);
  }, []);
  const publishR2rSource = useCallback((result: R2rSourceResult | null) => {
    setR2rSourceResult(result);
    setR2rResult(null);
  }, []);
  const publishAnalysisMotion = useCallback((motion: StageMotionPayload | null) => {
    setAnalysisMotion(motion);
    if (motion) setAnalysisRobotPreview(null);
  }, []);
  const publishAnalysisRobotPreview = useCallback(
    (preview: AnalysisRobotPreview | null) => {
      setAnalysisRobotPreview(preview);
      if (preview) setAnalysisMotion(null);
    },
    [],
  );

  // Workflow state is durable; this is the only projection into the shared Stage.
  useEffect(() => {
    if (activeView === "motion" || activeView === "video-to-motion") {
      setStageMotion(workspaceMotion);
      setStageRobot(null);
      setStageRobotTrajectory(null);
      setStageScaledMotion(null);
      return;
    }
    if (activeView === "robot-assets") {
      setStageMotion(null);
      setStageRobot(workspaceRobot);
      setStageRobotTrajectory(null);
      setStageScaledMotion(null);
      return;
    }
    if (activeView === "dataset-viz") {
      setStageMotion(analysisMotion);
      setStageRobot(analysisRobotPreview?.robot ?? null);
      setStageRobotTrajectory(analysisRobotPreview?.trajectory ?? null);
      setStageScaledMotion(
        analysisRobotPreview
          ? motionWithScene(
              null,
              analysisRobotPreview.scene,
              {
                kind: "dataset",
                token: analysisRobotPreview.previewToken,
              },
            )
          : null,
      );
      return;
    }
    if (activeView === "h2r") {
      const resultOwnsScaledSkeleton = Boolean(
        h2rResult && h2rResult.scaled_preview !== undefined,
      );
      const resultOwnsScaledPair = Boolean(
        h2rResult &&
          (resultOwnsScaledSkeleton ||
            h2rResult.scaled_scene !== undefined),
      );
      setStageMotion(h2rCalibrationReference ?? workspaceMotion);
      setStageRobot(workspaceRobot);
      setStageRobotTrajectory(
        h2rCalibrationPose
          ? calibrationTrajectory(h2rCalibrationPose, workspaceRobot)
          : h2rResult?.trajectory ?? null,
      );
      setStageScaledMotion(
        h2rCalibrationReference
          ? null
          : motionWithScene(
              resultOwnsScaledSkeleton
                ? h2rResult?.scaled_preview
                : h2rPreview?.preview,
              resultOwnsScaledPair
                ? h2rResult?.scaled_scene
                : h2rPreview?.scaled_scene,
              workspaceMotion?.token
                ? { kind: "motion", token: workspaceMotion.token }
                : undefined,
            ),
      );
      return;
    }
    if (activeView !== "r2r") {
      setStageMotion(null);
      setStageRobot(null);
      setStageRobotTrajectory(null);
      setStageScaledMotion(null);
      return;
    }

    // R2R is a symmetric two-actor presentation rendered through its own Stage
    // contract. Clear the single-actor slots so they cannot leak into it.
    setStageMotion(null);
    setStageRobot(null);
    setStageRobotTrajectory(null);
    setStageScaledMotion(null);
  }, [
    activeView,
    analysisMotion,
    analysisRobotPreview,
    h2rCalibrationPose,
    h2rCalibrationReference,
    h2rPreview,
    h2rResult,
    r2rCalibrationPose,
    r2rCalibrationReference,
    r2rResult,
    r2rSourceResult,
    r2rSourceRobot,
    r2rTargetRobot,
    workspaceMotion,
    workspaceRobot,
  ]);

  const sidebarHidden = tutorialOpen ? false : layout.sidebarHidden;
  const inspectorHidden = tutorialOpen ? false : layout.inspectorHidden;

  return (
    <LocaleProvider locale={locale}>
      <div
        id="app"
        className={`grid h-dvh min-h-0 min-w-0 ${coreWorkspaceReady ? "" : "invisible"}`}
        style={
          {
            "--workspace-sidebar-wide": sidebarHidden ? "0px" : "208px",
            "--workspace-sidebar-compact": sidebarHidden ? "0px" : "64px",
            "--workspace-inspector": inspectorHidden ? "0px" : "360px",
          } as CSSProperties
        }
        aria-busy={!coreWorkspaceReady}
        data-hhtools-ready={coreWorkspaceReady ? "true" : "false"}
        data-active-view={activeView}
        data-theme={theme}
        data-sidebar-hidden={sidebarHidden}
        data-inspector-hidden={inspectorHidden}
      >
      <Navbar
        locale={locale}
        theme={theme}
        canExportResult={currentExportUrl !== null}
        canExitApplication={Boolean(desktopBridge()?.exitApplication)}
        onNavigate={setActiveView}
        onImport={requestImport}
        onExportResult={() => exportLink.current?.click()}
        onOpenSettings={() => setDialog("settings")}
        onToggleTheme={toggleTheme}
        onOpenTutorial={openTutorial}
        onOpenAbout={() => setDialog("about")}
        onExitApplication={() => void desktopBridge()?.exitApplication?.()}
      />
      <Sidebar
        activeView={activeView}
        locale={locale}
        hidden={sidebarHidden}
        onSelect={setActiveView}
      />
      <Stage
        motion={stageMotion}
        scaledMotion={stageScaledMotion}
        robot={stageRobot}
        robotTrajectory={stageRobotTrajectory}
        presentation={stagePresentation}
        r2r={r2rStage}
        calibrationDisplay={
          activeView === "r2r" ? r2rCalibrationDisplay : h2rCalibrationDisplay
        }
        calibrationInteraction={
          activeView === "h2r"
            ? h2rCalibrationInteraction
            : activeView === "r2r"
              ? r2rCalibrationInteraction
              : null
        }
        layerPreset={
          activeView === "h2r" && h2rResult
            ? comparisonLayers("h2r", h2rComparisonPreset)
            : activeView === "r2r" && r2rResult
              ? comparisonLayers("r2r", r2rComparisonPreset)
              : null
        }
      />
      <Inspector hidden={inspectorHidden}>
        <div className={activeView === "motion" ? "h-full" : "hidden"}>
          <MotionView
            currentMotion={workspaceMotion}
            onMotionLoaded={publishMotion}
            onOpenSettings={
              desktopBridge()?.selectDirectory
                ? () => setDialog("settings")
                : undefined
            }
            importRequest={importRequest}
            libraryRevision={motionLibraryRevision}
          />
        </div>
        <div className={activeView === "robot-assets" ? "h-full" : "hidden"}>
          <RobotView
            currentRobot={workspaceRobot}
            onRobotLoaded={publishRobot}
            importRequest={importRequest}
          />
        </div>
        {(videoToMotionMounted || activeView === "video-to-motion") && (
          <div className={activeView === "video-to-motion" ? "h-full" : "hidden"}>
            <Suspense fallback={<VideoWorkspaceLoading />}>
              <VideoToMotionView
                onMotionLoaded={publishMotion}
                onMotionLibraryChange={noteMotionLibraryChanged}
                importRequest={importRequest}
                runtimeRevision={gvhmrRevision}
              />
            </Suspense>
          </div>
        )}
        <div className={activeView === "h2r" ? "h-full" : "hidden"}>
          <HumanToRobotView
            currentMotion={workspaceMotion}
            currentRobot={workspaceRobot}
            currentResult={h2rResult}
            onMotionLoaded={publishMotion}
            onRobotLoaded={publishRobot}
            onRetargetResult={setH2rResult}
            onCalibrationReference={setH2rCalibrationReference}
            onRobotPose={setH2rCalibrationPose}
            onScaledPreview={setH2rPreview}
            calibrationDisplay={h2rCalibrationDisplay}
            onCalibrationDisplayChange={setH2rCalibrationDisplay}
            onCalibrationInteraction={setH2rCalibrationInteraction}
            comparisonPreset={h2rComparisonPreset}
            onComparisonPresetChange={(preset) =>
              changeComparisonPreset("h2r", preset)
            }
            forceCalibrationOpen={
              tutorialOpen && tutorialStep === "calibration"
            }
            forceResultOpen={
              tutorialOpen &&
              (tutorialStep === "retarget" || tutorialStep === "export")
            }
            onOpenMotionLibrary={() => setActiveView("motion")}
            onOpenRobotLibrary={() => setActiveView("robot-assets")}
          />
        </div>
        <div className={activeView === "r2r" ? "h-full" : "hidden"}>
          <RobotToRobotView
            active={activeView === "r2r"}
            currentSourceRobot={r2rSourceRobot}
            currentTargetRobot={r2rTargetRobot}
            currentSourceResult={r2rSourceResult}
            currentResult={r2rResult}
            onSourceRobotLoaded={publishR2rSourceRobot}
            onTargetRobotLoaded={publishR2rTargetRobot}
            onSourceLoaded={publishR2rSource}
            onResultLoaded={setR2rResult}
            onCalibrationReference={setR2rCalibrationReference}
            onTargetPose={setR2rCalibrationPose}
            calibrationDisplay={r2rCalibrationDisplay}
            onCalibrationDisplayChange={setR2rCalibrationDisplay}
            onCalibrationInteraction={setR2rCalibrationInteraction}
            comparisonPreset={r2rComparisonPreset}
            onComparisonPresetChange={(preset) =>
              changeComparisonPreset("r2r", preset)
            }
            onOpenRobotLibrary={() => setActiveView("robot-assets")}
          />
        </div>
        <div className={activeView === "batch" ? "h-full" : "hidden"}>
          <BatchView
            active={activeView === "batch"}
            runtimeRevision={gvhmrRevision}
            humanEntries={humanBatchEntries}
            onHumanEntriesChange={setHumanBatchEntries}
            onMotionLibraryChange={noteMotionLibraryChanged}
          />
        </div>
        <div className={activeView === "dataset-viz" ? "h-full" : "hidden"}>
          <AnalysisView
            forceAnalysis={forceAnalysis}
            onMotionLoaded={publishAnalysisMotion}
            onRobotPreviewLoaded={publishAnalysisRobotPreview}
          />
        </div>
      </Inspector>
      <TaskDrawer />
      <a
        ref={exportLink}
        className="hidden"
        href={currentExportUrl ?? undefined}
        download
        aria-hidden="true"
      />
      <ApplicationDialogs
        dialog={dialog}
        locale={locale}
        sidebarHidden={layout.sidebarHidden}
        inspectorHidden={layout.inspectorHidden}
        forceAnalysis={forceAnalysis}
        onLocaleChange={changeLocale}
        onSidebarHiddenChange={(hidden) =>
          setLayout((current) => ({ ...current, sidebarHidden: hidden }))
        }
        onInspectorHiddenChange={(hidden) =>
          setLayout((current) => ({ ...current, inspectorHidden: hidden }))
        }
        onForceAnalysisChange={setForceAnalysis}
        onMotionLibraryChange={noteMotionLibraryChanged}
        onGvhmrChange={() =>
          setGvhmrRevision((revision) => revision + 1)
        }
        onClose={() => setDialog(null)}
      />
      <TutorialOverlay
        open={tutorialOpen}
        locale={locale}
        onNavigate={setActiveView}
        onClose={closeTutorial}
        onStepChange={changeTutorialStep}
      />
      </div>
    </LocaleProvider>
  );
}
