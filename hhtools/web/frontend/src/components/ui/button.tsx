import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex min-w-0 cursor-pointer items-center justify-center truncate whitespace-nowrap rounded-md text-xs font-medium tracking-normal transition-colors focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring disabled:cursor-not-allowed disabled:text-muted-foreground",
  {
    variants: {
      variant: {
        primary:
          "border border-primary bg-primary text-primary-foreground hover:bg-accent-foreground disabled:border-primary disabled:bg-primary disabled:text-primary-foreground disabled:opacity-50 disabled:hover:bg-primary",
        primaryOutline:
          "border border-primary/35 bg-surface text-primary hover:border-primary/60 hover:bg-accent disabled:border-primary/35 disabled:text-primary disabled:opacity-50 disabled:hover:bg-surface",
        danger:
          "border border-danger-border bg-danger-muted text-danger hover:border-danger hover:brightness-95 disabled:border-border disabled:bg-surface disabled:brightness-100",
        dangerOutline:
          "border border-danger-border bg-surface text-danger hover:border-danger/60 hover:bg-danger-muted disabled:border-danger-border disabled:bg-surface disabled:text-danger disabled:opacity-50",
        outline:
          "border border-border bg-surface text-foreground hover:border-primary hover:bg-accent disabled:hover:border-border disabled:hover:bg-surface",
        ghost:
          "border border-transparent bg-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground",
      },
      size: {
        default: "min-h-9 px-3.5 py-2",
        sm: "min-h-[30px] px-3 py-1.5",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
    },
  },
);

export function Button({
  className,
  variant,
  size,
  type = "button",
  ...props
}: ComponentProps<"button"> & VariantProps<typeof buttonVariants>) {
  return (
    <button
      type={type}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  );
}
