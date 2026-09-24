import { createContext, useCallback, useContext, type ReactNode } from "react";

import { localize, type WorkspaceLocale } from "@/localization";

const LocaleContext = createContext<WorkspaceLocale>("en");

export function LocaleProvider({
  locale,
  children,
}: {
  readonly locale: WorkspaceLocale;
  readonly children: ReactNode;
}) {
  return (
    <LocaleContext.Provider value={locale}>{children}</LocaleContext.Provider>
  );
}

export function useLocale(): WorkspaceLocale {
  return useContext(LocaleContext);
}

export function useLocaleText(): (
  english: string,
  chinese: string,
) => string {
  const locale = useLocale();
  return useCallback(
    (english, chinese) => localize(locale, english, chinese),
    [locale],
  );
}
