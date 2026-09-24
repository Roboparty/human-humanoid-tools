import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

interface SearchFieldProps extends ComponentProps<"input"> {
  label: string;
}

export function SearchField({
  label,
  className,
  type = "search",
  ...props
}: SearchFieldProps) {
  return (
    <label className="relative block min-w-0">
      <span className="sr-only">{label}</span>
      <span
        className="pointer-events-none absolute top-1/2 left-2.5 z-[1] size-3.5 -translate-y-1/2 bg-muted-foreground [mask:url(/icons/common/search.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/search.svg)_center/contain_no-repeat]"
        aria-hidden="true"
      />
      <input
        type={type}
        className={cn(
          "min-h-8 w-full rounded-md border border-border bg-surface py-1.5 pr-2.5 pl-8 text-xs text-foreground placeholder:text-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring disabled:cursor-not-allowed",
          className,
        )}
        {...props}
      />
    </label>
  );
}
