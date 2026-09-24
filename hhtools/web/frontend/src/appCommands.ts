import type { ViewId } from "./navigation";
import type { ApplicationImportTarget } from "./importIntent";
import { localize, type WorkspaceLocale } from "./localization.ts";

export type {
  ApplicationImportRequest,
  ApplicationImportTarget,
} from "./importIntent";

export type ApplicationTheme = "light" | "dark";
export type ApplicationMenuId =
  | "file"
  | "workflows"
  | "analysis"
  | "settings"
  | "help";
export type ApplicationCommandId =
  | "import-motion-file"
  | "import-motion-folder"
  | "import-video"
  | "import-robot-urdf"
  | "import-robot-mesh-folder"
  | "export-current-result"
  | "exit-application"
  | "navigate-video-to-motion"
  | "navigate-h2r"
  | "navigate-r2r"
  | "navigate-batch"
  | "navigate-analysis"
  | "open-settings"
  | "toggle-theme"
  | "open-tutorial"
  | "open-about";

export interface ApplicationCommand {
  readonly id: ApplicationCommandId;
  readonly label: string;
  readonly detail: string;
  readonly dividerBefore?: boolean;
  readonly enabled?: boolean;
  readonly disabledReason?: string;
  readonly run: () => void;
}

export interface ApplicationMenu {
  readonly id: ApplicationMenuId;
  readonly label: string;
  readonly commands: readonly ApplicationCommand[];
}

export interface ApplicationCommandContext {
  readonly locale: WorkspaceLocale;
  readonly theme: ApplicationTheme;
  readonly canExportResult: boolean;
  readonly canExitApplication: boolean;
  readonly onNavigate: (view: ViewId) => void;
  readonly onImport: (target: ApplicationImportTarget) => void;
  readonly onExportResult: () => void;
  readonly onOpenSettings: () => void;
  readonly onToggleTheme: () => void;
  readonly onOpenTutorial: () => void;
  readonly onOpenAbout: () => void;
  readonly onExitApplication: () => void;
}

const IMPORT_VIEWS: Readonly<Record<ApplicationImportTarget, ViewId>> = {
  "motion-file": "motion",
  "motion-folder": "motion",
  "video-file": "video-to-motion",
  "robot-urdf": "robot-assets",
  "robot-mesh-folder": "robot-assets",
};

export const THEME_STORAGE_KEY = "hhtools.theme";
export const PROJECT_README_URL =
  "https://github.com/Eleanor1018/human-humanoid-tools#readme";

interface ThemeStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function viewForImport(target: ApplicationImportTarget): ViewId {
  return IMPORT_VIEWS[target];
}

export function storedThemeOverride(
  storage: Pick<ThemeStorage, "getItem"> | undefined,
): ApplicationTheme | null {
  try {
    const stored = storage?.getItem(THEME_STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    return null;
  }
}

export function storedTheme(
  storage: Pick<ThemeStorage, "getItem"> | undefined,
  systemTheme: ApplicationTheme = "light",
): ApplicationTheme {
  return storedThemeOverride(storage) ?? systemTheme;
}

export function storeTheme(
  storage: Pick<ThemeStorage, "setItem"> | undefined,
  theme: ApplicationTheme,
): void {
  try {
    storage?.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Restricted browser contexts can reject persistence; live state remains valid.
  }
}

function navigationCommand(
  id: ApplicationCommandId,
  label: string,
  detail: string,
  view: ViewId,
  onNavigate: (view: ViewId) => void,
): ApplicationCommand {
  return {
    id,
    label,
    detail,
    run: () => onNavigate(view),
  };
}

function importCommand(
  id: ApplicationCommandId,
  label: string,
  detail: string,
  target: ApplicationImportTarget,
  onImport: (target: ApplicationImportTarget) => void,
  dividerBefore = false,
): ApplicationCommand {
  return {
    id,
    label,
    detail,
    dividerBefore,
    run: () => onImport(target),
  };
}

/** Build one typed command snapshot for the current App capabilities. */
export function createApplicationMenus(
  context: ApplicationCommandContext,
): readonly ApplicationMenu[] {
  const text = (english: string, chinese: string) =>
    localize(context.locale, english, chinese);
  const exitCommands: readonly ApplicationCommand[] = context.canExitApplication
    ? [
        {
          id: "exit-application",
          label: text("Exit", "退出"),
          detail: text("Close HHTOOLS", "关闭 HHTOOLS"),
          dividerBefore: true,
          run: context.onExitApplication,
        },
      ]
    : [];
  return [
    {
      id: "file",
      label: text("File", "文件"),
      commands: [
        importCommand(
          "import-motion-file",
          text("Import Motion File", "导入动作文件"),
          text("Import a motion asset", "导入一个动作资源"),
          "motion-file",
          context.onImport,
        ),
        importCommand(
          "import-motion-folder",
          text("Import Motion Folder", "导入动作文件夹"),
          text("Import a motion dataset folder", "导入一个动作数据集文件夹"),
          "motion-folder",
          context.onImport,
        ),
        importCommand(
          "import-video",
          text("Import Video", "导入视频"),
          text("Select a Video to Motion source", "选择视频转动作输入"),
          "video-file",
          context.onImport,
          true,
        ),
        importCommand(
          "import-robot-urdf",
          text("Import Robot URDF", "导入机器人 URDF"),
          text("Import a robot description", "导入机器人描述文件"),
          "robot-urdf",
          context.onImport,
          true,
        ),
        importCommand(
          "import-robot-mesh-folder",
          text("Import Robot Mesh Folder", "导入机器人网格文件夹"),
          text(
            "Select the meshes referenced by the robot URDF",
            "选择机器人 URDF 引用的网格文件",
          ),
          "robot-mesh-folder",
          context.onImport,
        ),
        {
          id: "export-current-result",
          label: text("Current Result…", "当前结果…"),
          detail: text(
            "Download the active retarget result as CSV",
            "以 CSV 下载当前重映射结果",
          ),
          enabled: context.canExportResult,
          disabledReason: context.canExportResult
            ? undefined
            : text("No exportable result", "没有可导出的结果"),
          run: context.onExportResult,
        },
        ...exitCommands,
      ],
    },
    {
      id: "workflows",
      label: text("Workflows", "工作流"),
      commands: [
        navigationCommand(
          "navigate-video-to-motion",
          text("Video to Motion", "视频转动作"),
          text("Generate motion from video with GVHMR", "使用 GVHMR 从视频生成动作"),
          "video-to-motion",
          context.onNavigate,
        ),
        navigationCommand(
          "navigate-h2r",
          text("Human to Robot", "人体转机器人"),
          text("Retarget human motion to a robot", "将人体动作重映射到机器人"),
          "h2r",
          context.onNavigate,
        ),
        navigationCommand(
          "navigate-r2r",
          text("Robot to Robot", "机器人转机器人"),
          text("Retarget a trajectory across robot embodiments", "在不同机器人之间重映射轨迹"),
          "r2r",
          context.onNavigate,
        ),
        navigationCommand(
          "navigate-batch",
          text("Batch", "批处理"),
          text("Run batch workflows", "运行批处理工作流"),
          "batch",
          context.onNavigate,
        ),
      ],
    },
    {
      id: "analysis",
      label: text("Analysis", "分析"),
      commands: [
        navigationCommand(
          "navigate-analysis",
          text("Data Analysis", "数据分析"),
          text("Inspect motion and trajectory datasets", "检查动作与轨迹数据集"),
          "dataset-viz",
          context.onNavigate,
        ),
      ],
    },
    {
      id: "settings",
      label: text("Settings", "设置"),
      commands: [
        {
          id: "open-settings",
          label: text("Settings", "设置"),
          detail: text(
            "Configure workspace and background jobs",
            "配置工作区与后台任务",
          ),
          run: context.onOpenSettings,
        },
        {
          id: "toggle-theme",
          label:
            context.theme === "dark"
              ? text("Light Mode", "浅色模式")
              : text("Dark Mode", "深色模式"),
          detail:
            context.theme === "dark"
              ? text("Switch to light appearance", "切换到浅色外观")
              : text("Switch to dark appearance", "切换到深色外观"),
          run: context.onToggleTheme,
        },
      ],
    },
    {
      id: "help",
      label: text("Help", "帮助"),
      commands: [
        {
          id: "open-tutorial",
          label: text("Tutorial", "教程"),
          detail: text(
            "Open the interactive workspace guide",
            "打开交互式工作区教程",
          ),
          run: context.onOpenTutorial,
        },
        {
          id: "open-about",
          label: text("About hhtools", "关于 hhtools"),
          detail: text("Project and source information", "项目与源代码信息"),
          dividerBefore: true,
          run: context.onOpenAbout,
        },
      ],
    },
  ];
}
