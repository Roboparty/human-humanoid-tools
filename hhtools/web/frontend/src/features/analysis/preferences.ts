export const FORCE_REANALYSIS_STORAGE_KEY =
  "hhtools.analysis.force-reanalysis";

interface PreferenceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function storedForceReanalysis(
  storage: Pick<PreferenceStorage, "getItem"> | undefined,
): boolean {
  try {
    return storage?.getItem(FORCE_REANALYSIS_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

export function storeForceReanalysis(
  storage: Pick<PreferenceStorage, "setItem"> | undefined,
  force: boolean,
): void {
  try {
    storage?.setItem(FORCE_REANALYSIS_STORAGE_KEY, String(force));
  } catch {
    // Restricted browser contexts can reject persistence; live state still works.
  }
}
