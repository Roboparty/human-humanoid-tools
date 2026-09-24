export type WorkspaceLocale = "en" | "zh-CN";

export const WORKSPACE_LOCALE_STORAGE_KEY =
  "hhtools-workspace-preferences-v1";

interface LocaleStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function systemLocale(tags: readonly string[]): WorkspaceLocale {
  for (const tag of tags) {
    const language = tag.trim().toLowerCase().split(/[-_]/, 1)[0];
    if (language === "zh") return "zh-CN";
    if (language === "en") return "en";
  }
  return "en";
}

export function localize(
  locale: WorkspaceLocale,
  english: string,
  chinese: string,
): string {
  return locale === "zh-CN" ? chinese : english;
}

export function storedLocaleOverride(
  storage: Pick<LocaleStorage, "getItem"> | undefined,
): WorkspaceLocale | null {
  try {
    const value = JSON.parse(
      storage?.getItem(WORKSPACE_LOCALE_STORAGE_KEY) ?? "{}",
    ) as { readonly locale?: unknown };
    if (value.locale === "en" || value.locale === "zh-CN") {
      return value.locale;
    }
  } catch {
    // A malformed legacy preference must not block the workbench.
  }
  return null;
}

export function storedLocale(
  storage: Pick<LocaleStorage, "getItem"> | undefined,
  languageTags: readonly string[] = [],
): WorkspaceLocale {
  return storedLocaleOverride(storage) ?? systemLocale(languageTags);
}

export function storeLocale(
  storage: LocaleStorage | undefined,
  locale: WorkspaceLocale,
): void {
  if (!storage) return;
  try {
    let current: Record<string, unknown> = {};
    try {
      const parsed = JSON.parse(
        storage.getItem(WORKSPACE_LOCALE_STORAGE_KEY) ?? "{}",
      ) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        current = parsed as Record<string, unknown>;
      }
    } catch {
      // Replace malformed legacy preferences with one valid locale value.
    }
    storage.setItem(
      WORKSPACE_LOCALE_STORAGE_KEY,
      JSON.stringify({ ...current, locale }),
    );
  } catch {
    // Restricted browser contexts can reject persistence; live state remains valid.
  }
}
