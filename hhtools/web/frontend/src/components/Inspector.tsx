import type { ReactNode } from "react";
import { useLocaleText } from "@/LocaleProvider";

export function Inspector({
  children,
  hidden = false,
}: {
  children: ReactNode;
  hidden?: boolean;
}) {
  const text = useLocaleText();
  return (
    <aside
      hidden={hidden}
      className="col-start-3 row-start-2 row-span-2 min-h-0 min-w-0 overflow-hidden border-l border-border-subtle bg-surface max-[780px]:col-start-2 max-[780px]:row-start-3 max-[780px]:row-span-1 max-[780px]:border-t"
      aria-label={text("Inspector", "检查器")}
    >
      {children}
    </aside>
  );
}

export function InspectorPage({
  title,
  headerAction,
  children,
}: {
  title: string;
  headerAction?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section
      className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto p-[18px]"
      aria-label={title}
    >
      <header className="flex min-h-[30px] items-center justify-between gap-3">
        <h1 className="text-[19px] leading-tight font-bold tracking-normal text-foreground">
          {title}
        </h1>
        {headerAction}
      </header>
      {children}
    </section>
  );
}
