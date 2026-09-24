import type { ReactNode } from "react";

export const fieldClass =
  "min-h-8 w-full min-w-0 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring disabled:cursor-not-allowed disabled:text-muted-foreground";

export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="grid min-w-0 gap-1.5 text-xs font-medium text-foreground">
      {label}
      {children}
    </label>
  );
}
