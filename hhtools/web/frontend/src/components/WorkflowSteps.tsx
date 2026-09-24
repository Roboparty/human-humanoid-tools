import { useState, type CSSProperties, type ReactNode } from "react";

import { useLocaleText } from "@/LocaleProvider";
import { cn } from "@/lib/utils";
import {
  workflowPipelineState,
  workflowStatusToneClass,
  type WorkflowStatusTone,
} from "./workflowPipeline";

export type { WorkflowStatusTone } from "./workflowPipeline";

interface WorkflowPipelineProps {
  label: string;
  steps: readonly string[];
  activeIndex?: number;
  completedIndex?: number;
}

export function WorkflowPipeline({
  label,
  steps,
  activeIndex = 0,
  completedIndex = activeIndex - 1,
}: WorkflowPipelineProps) {
  const text = useLocaleText();
  return (
    <ol
      className="grid min-h-[54px] shrink-0 gap-0"
      style={
        { gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` } as CSSProperties
      }
      aria-label={label}
    >
      {steps.map((step, index) => {
        const state = workflowPipelineState(index, activeIndex, completedIndex);
        return (
          <li
            key={step}
            className="relative flex min-w-0 flex-col items-center gap-1.5 text-center"
            data-state={state}
            aria-label={`${step}, ${
              state === "active"
                ? text("active", "当前")
                : state === "complete"
                  ? text("complete", "已完成")
                  : text("upcoming", "未开始")
            }`}
            aria-current={state === "active" ? "step" : undefined}
          >
            {index > 0 && (
              <span
                className={cn(
                  "absolute top-[5px] right-1/2 h-px w-full bg-border-subtle",
                  index <= completedIndex + 1 && "bg-success",
                )}
                aria-hidden="true"
              />
            )}
            <span
              className={cn(
                "relative z-[1] size-2.5 rounded-full border-2 border-surface bg-border",
                state === "active" && "bg-primary",
                state === "complete" && "bg-success",
              )}
              aria-hidden="true"
            />
            <span
              className={cn(
                "max-w-full px-1 text-[11px] leading-tight text-muted-foreground",
                state === "active" && "font-semibold text-primary",
                state === "complete" && "font-semibold text-success",
              )}
            >
              {step}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

interface WorkflowStepProps {
  title: string;
  status?: string;
  statusTone?: WorkflowStatusTone;
  defaultOpen?: boolean;
  forceOpen?: boolean;
  tutorialAnchor?: string;
  children: ReactNode;
}

export function WorkflowStep({
  title,
  status,
  statusTone = "neutral",
  defaultOpen = false,
  forceOpen = false,
  tutorialAnchor,
  children,
}: WorkflowStepProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details
      className="group border-b border-border-subtle"
      data-tutorial={tutorialAnchor}
      open={forceOpen || open}
    >
      <summary
        className="flex min-h-[42px] cursor-pointer list-none items-center gap-2 text-[13px] font-semibold text-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring [&::-webkit-details-marker]:hidden"
        onClick={(event) => {
          event.preventDefault();
          if (!forceOpen) setOpen((current) => !current);
        }}
      >
        <span className="min-w-0 flex-1 truncate">{title}</span>
        {status && (
          <span
            className={cn(
              "max-w-[50%] shrink-0 truncate text-[11px] font-normal",
              workflowStatusToneClass(statusTone),
            )}
            data-status-tone={statusTone}
            title={status}
          >
            {status}
          </span>
        )}
        <span
          className="size-4 shrink-0 bg-muted-foreground transition-transform [mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/chevron-down.svg)_center/contain_no-repeat] group-open:rotate-180"
          aria-hidden="true"
        />
      </summary>
      <div className="pt-0.5 pb-4">{children}</div>
    </details>
  );
}
