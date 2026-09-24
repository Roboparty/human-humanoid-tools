import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { Field, fieldClass } from "@/components/Field";
import { RefreshButton } from "@/components/RefreshButton";
import { Button } from "@/components/ui/button";
import {
  desktopSettingsBridge,
  getGvhmrRuntimeSettings,
  getJobAdmissionSettings,
  getMotionLibrarySettings,
  invalidateGvhmrRuntimeStatus,
  updateJobAdmissionSettings,
  updateMotionLibrarySettings,
  type GvhmrOptionalComponentState,
  type GvhmrRuntimeStatus,
  type JobAdmissionSnapshot,
  type MotionLibrarySettingsSnapshot,
} from "@/features/settings/api";
import { localize, type WorkspaceLocale } from "@/localization";
import { SmplxModelLinks } from "@/features/video-to-motion/SmplxModelLinks";

export type ApplicationDialog = "settings" | "about" | null;

export interface ApplicationDialogsProps {
  readonly dialog: ApplicationDialog;
  readonly locale: WorkspaceLocale;
  readonly sidebarHidden: boolean;
  readonly inspectorHidden: boolean;
  readonly forceAnalysis: boolean;
  readonly onLocaleChange: (locale: WorkspaceLocale) => void;
  readonly onSidebarHiddenChange: (hidden: boolean) => void;
  readonly onInspectorHiddenChange: (hidden: boolean) => void;
  readonly onForceAnalysisChange: (force: boolean) => void;
  readonly onMotionLibraryChange: (
    settings: MotionLibrarySettingsSnapshot,
  ) => void;
  readonly onGvhmrChange: (status: GvhmrRuntimeStatus) => void;
  readonly onClose: () => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function Modal({
  title,
  closeLabel,
  wide = false,
  onClose,
  children,
}: {
  readonly title: string;
  readonly closeLabel: string;
  readonly wide?: boolean;
  readonly onClose: () => void;
  readonly children: ReactNode;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    panel.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[400] grid place-items-center bg-black/35 p-4"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={panel}
        className={`grid max-h-[88vh] w-full gap-4 overflow-y-auto rounded-lg border border-border-subtle bg-surface p-4 text-foreground shadow-[0_18px_50px_rgba(0,0,0,0.22)] outline-none ${wide ? "max-w-[640px]" : "max-w-[480px]"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="application-dialog-title"
        tabIndex={-1}
      >
        <header className="flex items-center justify-between gap-3">
          <h2
            id="application-dialog-title"
            className="text-base font-bold tracking-normal"
          >
            {title}
          </h2>
          <Button size="sm" variant="ghost" onClick={onClose}>
            <span
              className="size-4 bg-current [mask:url(/icons/common/close.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/close.svg)_center/contain_no-repeat]"
              aria-hidden="true"
            />
            <span className="sr-only">{closeLabel}</span>
          </Button>
        </header>
        {children}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { readonly children: ReactNode }) {
  return (
    <h3 className="border-t border-border-subtle pt-3 text-xs font-semibold text-foreground first:border-0 first:pt-0">
      {children}
    </h3>
  );
}

function SettingRow({
  title,
  detail,
  children,
}: {
  readonly title: string;
  readonly detail?: string;
  readonly children: ReactNode;
}) {
  return (
    <div className="flex min-h-10 items-center justify-between gap-4 py-1">
      <span className="min-w-0">
        <strong className="block text-xs font-medium text-foreground">
          {title}
        </strong>
        {detail ? (
          <small className="block text-[11px] leading-relaxed text-muted-foreground">
            {detail}
          </small>
        ) : null}
      </span>
      <span className="shrink-0">{children}</span>
    </div>
  );
}

function nonNegativeInteger(value: string): number | null {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
}

interface SettingsDialogProps
  extends Omit<ApplicationDialogsProps, "dialog" | "onClose"> {
  readonly onClose: () => void;
}

function SettingsDialog(props: SettingsDialogProps) {
  const text = (english: string, chinese: string) =>
    localize(props.locale, english, chinese);
  const [jobAdmission, setJobAdmission] =
    useState<JobAdmissionSnapshot | null>(null);
  const [motionLibrary, setMotionLibrary] =
    useState<MotionLibrarySettingsSnapshot | null>(null);
  const [gvhmrComponent, setGvhmrComponent] =
    useState<GvhmrOptionalComponentState | null>(null);
  const [gvhmrRuntime, setGvhmrRuntime] =
    useState<GvhmrRuntimeStatus | null>(null);
  const [runningLimit, setRunningLimit] = useState("0");
  const [queueLimit, setQueueLimit] = useState("0");
  const [batchItemLimit, setBatchItemLimit] = useState("0");
  const [batchFrameLimit, setBatchFrameLimit] = useState("0");
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState<"jobs" | "library" | "gvhmr" | null>(
    null,
  );
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refreshRequest = useRef<AbortController | null>(null);

  const refresh = useCallback(async (freshGvhmr = false) => {
    if (freshGvhmr) invalidateGvhmrRuntimeStatus();
    refreshRequest.current?.abort();
    const request = new AbortController();
    refreshRequest.current = request;
    setLoading(true);
    setSaved(false);
    setError(null);
    try {
      const desktop = desktopSettingsBridge();
      const [jobs, library, runtime, components] = await Promise.all([
        getJobAdmissionSettings({ signal: request.signal }),
        getMotionLibrarySettings({ signal: request.signal }),
        getGvhmrRuntimeSettings({ signal: request.signal }).catch(() => null),
        desktop?.getOptionalComponents() ?? null,
      ]);
      if (request.signal.aborted) return null;
      setJobAdmission(jobs);
      setRunningLimit(String(jobs.max_running_jobs));
      setQueueLimit(String(jobs.max_queued_jobs));
      setBatchItemLimit(String(jobs.max_batch_items));
      setBatchFrameLimit(String(jobs.max_batch_total_frames));
      setMotionLibrary(library);
      setGvhmrRuntime(runtime);
      setGvhmrComponent(components?.gvhmr ?? null);
      return runtime;
    } catch (reason) {
      if (!request.signal.aborted) setError(errorMessage(reason));
      return null;
    } finally {
      if (!request.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => refreshRequest.current?.abort();
  }, [refresh]);

  const parsedRunning = nonNegativeInteger(runningLimit);
  const parsedQueued = nonNegativeInteger(queueLimit);
  const parsedBatchItems = nonNegativeInteger(batchItemLimit);
  const parsedBatchFrames = nonNegativeInteger(batchFrameLimit);
  const limitsValid =
    parsedRunning !== null &&
    parsedQueued !== null &&
    parsedBatchItems !== null &&
    parsedBatchFrames !== null;
  const busy = loading || action !== null;

  const saveJobs = async () => {
    if (
      !limitsValid ||
      parsedRunning === null ||
      parsedQueued === null ||
      parsedBatchItems === null ||
      parsedBatchFrames === null ||
      jobAdmission?.editable !== true
    ) {
      return;
    }
    setAction("jobs");
    setSaved(false);
    setError(null);
    try {
      const result = await updateJobAdmissionSettings({
        max_running_jobs: parsedRunning,
        max_queued_jobs: parsedQueued,
        max_batch_items: parsedBatchItems,
        max_batch_total_frames: parsedBatchFrames,
      });
      setJobAdmission(result);
      setRunningLimit(String(result.max_running_jobs));
      setQueueLimit(String(result.max_queued_jobs));
      setBatchItemLimit(String(result.max_batch_items));
      setBatchFrameLimit(String(result.max_batch_total_frames));
      setSaved(true);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setAction(null);
    }
  };

  const chooseMotionLibrary = async () => {
    if (motionLibrary?.editable !== true) return;
    setAction("library");
    setSaved(false);
    setError(null);
    try {
      const desktop = desktopSettingsBridge();
      if (!desktop) return;
      const selected = await desktop.selectDirectory();
      if (!selected?.trim()) return;
      const result = await updateMotionLibrarySettings(selected.trim());
      setMotionLibrary(result);
      props.onMotionLibraryChange(result);
      setSaved(true);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setAction(null);
    }
  };

  const setupGvhmr = async () => {
    const desktop = desktopSettingsBridge();
    if (!desktop) return;
    setAction("gvhmr");
    setSaved(false);
    setError(null);
    try {
      const result = await desktop.setupGvhmr();
      setGvhmrComponent(result.state);
      if (result.action === "cancelled") return;
      const runtime = await refresh(true);
      if (runtime) props.onGvhmrChange(runtime);
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setAction(null);
    }
  };

  return (
    <Modal
      title={text("Workspace Settings", "工作区设置")}
      closeLabel={text("Close", "关闭")}
      wide
      onClose={props.onClose}
    >
      <div className="grid gap-3">
        <SectionTitle>{text("Workspace", "工作区")}</SectionTitle>
        <SettingRow title={text("Language", "语言")}>
          <select
            className={`${fieldClass} min-w-[126px]`}
            aria-label={text("Workspace language", "工作区语言")}
            value={props.locale}
            onChange={(event) =>
              props.onLocaleChange(event.currentTarget.value as WorkspaceLocale)
            }
          >
            <option value="en">English</option>
            <option value="zh-CN">简体中文</option>
          </select>
        </SettingRow>
        <SettingRow title={text("Left navigation", "左侧导航")}>
          <input
            className="size-4 accent-primary"
            type="checkbox"
            aria-label={text("Show left navigation", "显示左侧导航")}
            checked={!props.sidebarHidden}
            onChange={(event) =>
              props.onSidebarHiddenChange(!event.currentTarget.checked)
            }
          />
        </SettingRow>
        <SettingRow title={text("Right inspector", "右侧控制面板")}>
          <input
            className="size-4 accent-primary"
            type="checkbox"
            aria-label={text("Show right inspector", "显示右侧控制面板")}
            checked={!props.inspectorHidden}
            onChange={(event) =>
              props.onInspectorHiddenChange(!event.currentTarget.checked)
            }
          />
        </SettingRow>

        {desktopSettingsBridge() && (
          <>
            <SectionTitle>{text("Motion library", "动作资源库")}</SectionTitle>
            <SettingRow title={text("Library directory", "资源库目录")}>
              <Button
                size="sm"
                disabled={busy || motionLibrary?.editable !== true}
                onClick={() => void chooseMotionLibrary()}
              >
                {action === "library"
                  ? text("Choosing…", "选择中……")
                  : text("Choose directory", "选择目录")}
              </Button>
            </SettingRow>
          </>
        )}

        <SectionTitle>{text("Optional components", "可选组件")}</SectionTitle>
        <SettingRow
          title={text("GVHMR video-to-motion", "GVHMR 视频转动作")}
          detail={
            gvhmrRuntime?.ready
              ? text("Ready", "已就绪")
              : gvhmrRuntime?.missing.slice(0, 2).join(" · ") ||
                text("Needs configuration", "需要配置")
          }
        >
          {gvhmrComponent ? (
              <Button
                size="sm"
                disabled={busy}
                onClick={() => void setupGvhmr()}
              >
                {action === "gvhmr"
                  ? text("Setting up…", "配置中……")
                  : text("Set up / repair", "配置 / 修复")}
              </Button>
          ) : (
            <span className="text-[11px] text-muted-foreground">
              {text("Desktop setup only", "仅桌面端可配置")}
            </span>
          )}
        </SettingRow>
        <SmplxModelLinks runtime={gvhmrRuntime} />

        <SectionTitle>{text("Analysis", "分析")}</SectionTitle>
        <SettingRow title={text("Force re-analysis", "强制重新分析")}>
          <input
            className="size-4 accent-primary"
            type="checkbox"
            aria-label={text("Force re-analysis", "强制重新分析")}
            checked={props.forceAnalysis}
            onChange={(event) =>
              props.onForceAnalysisChange(event.currentTarget.checked)
            }
          />
        </SettingRow>

        <SectionTitle>
          {text("Background-job scheduling", "后台任务调度")}
        </SectionTitle>
        <div className="grid grid-cols-2 gap-3 max-[520px]:grid-cols-1">
          <Field label={text("Maximum running jobs", "最大并发任务数")}>
            <input
              className={fieldClass}
              type="number"
              min="0"
              step="1"
              value={runningLimit}
              disabled={busy || jobAdmission?.editable !== true}
              onChange={(event) => setRunningLimit(event.currentTarget.value)}
            />
          </Field>
          <Field label={text("Maximum queued jobs", "最大等待任务数")}>
            <input
              className={fieldClass}
              type="number"
              min="0"
              step="1"
              value={queueLimit}
              disabled={busy || jobAdmission?.editable !== true}
              onChange={(event) => setQueueLimit(event.currentTarget.value)}
            />
          </Field>
          <Field label={text("Maximum batch items", "单批最大条目数")}>
            <input
              className={fieldClass}
              type="number"
              min="0"
              step="1"
              value={batchItemLimit}
              disabled={busy || jobAdmission?.editable !== true}
              onChange={(event) => setBatchItemLimit(event.currentTarget.value)}
            />
          </Field>
          <Field label={text("Maximum batch frames", "单批最大总帧数")}>
            <input
              className={fieldClass}
              type="number"
              min="0"
              step="1"
              value={batchFrameLimit}
              disabled={busy || jobAdmission?.editable !== true}
              onChange={(event) => setBatchFrameLimit(event.currentTarget.value)}
            />
          </Field>
        </div>
        <p className="text-[11px] text-muted-foreground">
          {text(
            "Use 0 for unlimited. Batch processing remains serial within each job.",
            "设为 0 表示不限；单个批任务内部仍按顺序执行。",
          )}
        </p>
        {!limitsValid ? (
          <p className="text-[11px] text-danger" role="alert">
            {text(
              "Enter integers greater than or equal to 0.",
              "请输入 0 或更大的整数。",
            )}
          </p>
        ) : null}
        {jobAdmission?.editable === false ? (
          <p className="text-[11px] text-muted-foreground">
            {text(
              "These limits can only be changed from the local WebUI or desktop app.",
              "这些限制只能在本机 WebUI 或桌面应用中修改。",
            )}
          </p>
        ) : null}
        {error ? (
          <p
            className="rounded-md border border-danger-border bg-danger-muted px-2.5 py-2 text-[11px] text-danger"
            role="alert"
          >
            {error}
          </p>
        ) : null}
        {saved && !error ? (
          <p className="text-[11px] font-medium text-success" role="status">
            {text("Settings saved.", "设置已保存。")}
          </p>
        ) : null}
      </div>

      <footer className="flex items-center justify-between gap-3 border-t border-border-subtle pt-3">
        <RefreshButton
          variant="ghost"
          label={text("Refresh settings", "刷新设置")}
          busy={loading}
          disabled={action !== null}
          onClick={() => void refresh(true)}
        />
        <Button
          size="sm"
          variant="primary"
          disabled={busy || !limitsValid || jobAdmission?.editable !== true}
          onClick={() => void saveJobs()}
        >
          {action === "jobs"
            ? text("Saving…", "保存中……")
            : text("Save", "保存")}
        </Button>
      </footer>
    </Modal>
  );
}

function AboutDialog({
  locale,
  onClose,
}: {
  readonly locale: WorkspaceLocale;
  readonly onClose: () => void;
}) {
  const text = (english: string, chinese: string) =>
    localize(locale, english, chinese);
  return (
    <Modal
      title="Human-Humanoid Tools"
      closeLabel={text("Close", "关闭")}
      onClose={onClose}
    >
      <div className="grid gap-4 text-sm leading-relaxed text-muted-foreground">
        <img
          className="h-auto w-[168px] [html[data-theme=dark]_&]:brightness-0 [html[data-theme=dark]_&]:invert"
          src="/roboparty.svg"
          alt="ROBOPARTY"
        />
        <p>
          {text(
            "Humanoid motion retargeting and dataset analysis",
            "人形机器人动作重映射与数据集分析工具",
          )}
        </p>
        <dl className="grid gap-2">
          <div>
            <dt className="text-xs font-medium text-foreground">
              {text("Authors and contributors", "作者与贡献者")}
            </dt>
            <dd>
              {text(
                "Jagger Shen, Nora Sun and hhtools contributors",
                "Jagger Shen、Nora Sun 与 hhtools 贡献者",
              )}
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-foreground">
              {text("Year", "年份")}
            </dt>
            <dd>2026</dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-foreground">
              {text("Source code", "源代码")}
            </dt>
            <dd>
              <a
                className="text-primary hover:underline"
                href="https://github.com/Roboparty/human-humanoid-tools"
                target="_blank"
                rel="noreferrer"
              >
                github.com/Roboparty/human-humanoid-tools
              </a>
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-foreground">
              {text("License", "许可证")}
            </dt>
            <dd>Apache-2.0</dd>
          </div>
        </dl>
        <section aria-label={text("Contact", "联系")}>
          <h3 className="text-xs font-medium text-foreground">
            {text("Contact", "联系")}
          </h3>
          <div className="flex flex-col items-start">
            <a
              className="text-primary hover:underline"
              href="mailto:shenyaojie@roboparty.com"
            >
              shenyaojie@roboparty.com
            </a>
            <a
              className="text-primary hover:underline"
              href="mailto:sunlancheng@roboparty.com"
            >
              sunlancheng@roboparty.com
            </a>
          </div>
        </section>
      </div>
    </Modal>
  );
}

export function ApplicationDialogs(props: ApplicationDialogsProps) {
  if (props.dialog === "settings") {
    return <SettingsDialog {...props} />;
  }
  if (props.dialog === "about") {
    return <AboutDialog locale={props.locale} onClose={props.onClose} />;
  }
  return null;
}
