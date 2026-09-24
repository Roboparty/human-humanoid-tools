export interface ExecutionProvenance {
  readonly executor: string;
  readonly backend: string;
  readonly device: string;
  readonly device_kind: "cpu" | "cuda" | "mps" | "unknown";
  readonly precision: "float32" | "float64" | "mixed" | "unknown";
  readonly runtime?: string;
  readonly runtime_version?: string;
  readonly solver?: string;
  readonly cuda_runtime?: string;
  readonly cuda_graph_requested?: boolean;
  readonly cuda_graph_used?: boolean;
  readonly fallback_used: boolean;
  readonly fallback_reason?: string;
}
