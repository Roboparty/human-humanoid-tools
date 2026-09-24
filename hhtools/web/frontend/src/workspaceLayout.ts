export interface WorkspaceLayout {
  readonly sidebarHidden: boolean;
  readonly inspectorHidden: boolean;
}

export const DEFAULT_WORKSPACE_LAYOUT: WorkspaceLayout = {
  sidebarHidden: false,
  inspectorHidden: false,
};

export const WORKSPACE_LAYOUT_STORAGE_KEY =
  "hhtools-desktop-panel-layout-v1";

interface LayoutStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function storedWorkspaceLayout(
  storage: Pick<LayoutStorage, "getItem"> | undefined,
): WorkspaceLayout {
  try {
    const value = JSON.parse(
      storage?.getItem(WORKSPACE_LAYOUT_STORAGE_KEY) ?? "{}",
    ) as Readonly<Record<string, unknown>>;
    return {
      sidebarHidden: value.sidebarHidden === true,
      inspectorHidden: value.inspectorHidden === true,
    };
  } catch {
    return DEFAULT_WORKSPACE_LAYOUT;
  }
}

export function storeWorkspaceLayout(
  storage: LayoutStorage | undefined,
  layout: WorkspaceLayout,
): void {
  try {
    storage?.setItem(WORKSPACE_LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // Restricted browser contexts can reject persistence; live state remains valid.
  }
}
