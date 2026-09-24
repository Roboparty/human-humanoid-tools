import type { ComponentProps } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function RefreshButton({
  busy = false,
  label = "Refresh",
  className,
  disabled,
  ...props
}: Omit<ComponentProps<typeof Button>, "children" | "size"> & {
  readonly busy?: boolean;
  readonly label?: string;
}) {
  return (
    <Button
      {...props}
      size="sm"
      className={cn("size-[30px] min-h-[30px] shrink-0 p-0", className)}
      disabled={disabled || busy}
      aria-busy={busy}
      aria-label={label}
      title={label}
    >
      <span
        className={cn(
          "size-4 bg-current [mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/refresh-cw.svg)_center/contain_no-repeat]",
          busy && "animate-spin",
        )}
        aria-hidden="true"
      />
    </Button>
  );
}
