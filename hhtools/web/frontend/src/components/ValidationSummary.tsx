import { cn } from "@/lib/utils";

export type ValidationTone = "ok" | "warning" | "error" | "neutral";

export interface ValidationItem {
  readonly tone: ValidationTone;
  readonly label: string;
}

const toneClass: Readonly<Record<ValidationTone, string>> = {
  ok: "bg-success",
  warning: "bg-warning",
  error: "bg-danger",
  neutral: "bg-muted-foreground",
};

/** Compact, unframed facts derived from the current authoritative payload. */
export function ValidationSummary({
  items,
  label = "Validation",
}: {
  readonly items: readonly ValidationItem[];
  readonly label?: string;
}) {
  if (!items.length) return null;
  return (
    <section className="grid gap-1.5 text-[11px]" aria-label={label}>
      {items.map((item, index) => (
        <div
          key={`${item.label}-${index}`}
          className="grid grid-cols-[8px_minmax(0,1fr)] items-start gap-2 leading-[1.4] text-muted-foreground"
        >
          <span
            className={cn("mt-[4px] size-2 rounded-full", toneClass[item.tone])}
            aria-hidden="true"
          />
          <span className="min-w-0 break-words">{item.label}</span>
        </div>
      ))}
    </section>
  );
}
