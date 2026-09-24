export type WorkflowPipelineState = "complete" | "active" | "pending";
export type WorkflowStatusTone =
  | "neutral"
  | "info"
  | "success"
  | "warning"
  | "danger";

const STATUS_TONE_CLASS: Readonly<Record<WorkflowStatusTone, string>> = {
  neutral: "text-muted-foreground",
  info: "text-primary",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

export function workflowStatusToneClass(tone: WorkflowStatusTone): string {
  return STATUS_TONE_CLASS[tone];
}

export function workflowPipelineState(
  index: number,
  activeIndex: number,
  completedIndex: number,
): WorkflowPipelineState {
  if (index <= completedIndex) return "complete";
  if (index === activeIndex) return "active";
  return "pending";
}
