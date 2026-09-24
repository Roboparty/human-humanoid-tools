import { useState } from "react";

import { Field, fieldClass } from "@/components/Field";
import { Button } from "@/components/ui/button";
import { useLocaleText } from "@/LocaleProvider";

import {
  validateExportOptions,
  type ExportFormat,
  type ExportOptions,
} from "./model";

function localizedValidationError(
  error: string,
  text: (english: string, chinese: string) => string,
): string {
  const translations: Readonly<Record<string, string>> = {
    "Export FPS must be greater than zero.": "导出 FPS 必须大于零。",
    "Start time must be zero or greater.": "开始时间必须大于或等于零。",
    "End time must be zero or greater.": "结束时间必须大于或等于零。",
    "End time must be greater than start time.": "结束时间必须晚于开始时间。",
  };
  return text(error, translations[error] ?? error);
}

export function ResultExportControls({
  token,
  resultFps,
  hasScene = false,
  buildUrl,
}: {
  readonly token: string;
  readonly resultFps?: number;
  readonly hasScene?: boolean;
  readonly buildUrl: (token: string, options: ExportOptions) => string;
}) {
  const text = useLocaleText();
  const [format, setFormat] = useState<ExportFormat>("csv");
  const [fps, setFps] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [csvHeader, setCsvHeader] = useState(true);

  const validation = validateExportOptions({
    format,
    fps,
    start,
    end,
    csvHeader,
  });
  const exportUrl = validation.valid
    ? buildUrl(token, validation.options)
    : null;
  const defaultFps =
    typeof resultFps === "number" && Number.isFinite(resultFps)
      ? resultFps
      : null;

  return (
    <section
      className="grid gap-2.5 border-t border-border-subtle pt-3"
      aria-label={text("Export result", "导出结果")}
    >
      <div className="grid grid-cols-2 gap-2">
        <Field label={text("Export FPS", "导出 FPS")}>
          <input
            className={fieldClass}
            type="number"
            min="0.001"
            step="any"
            placeholder={
              defaultFps
                ? text(
                    `Result: ${defaultFps.toFixed(1)}`,
                    `结果：${defaultFps.toFixed(1)}`,
                  )
                : text("Result FPS", "结果 FPS")
            }
            value={fps}
            aria-invalid={
              !validation.valid && validation.error.startsWith("Export FPS")
            }
            onChange={(event) => setFps(event.currentTarget.value)}
          />
        </Field>
        <Field label={text("Format", "格式")}>
          <select
            className={fieldClass}
            value={format}
            onChange={(event) =>
              setFormat(event.currentTarget.value as ExportFormat)
            }
          >
            <option value="csv">CSV</option>
            <option value="pkl">PKL</option>
          </select>
        </Field>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Field label={text("Start (s)", "开始（秒）")}>
          <input
            className={fieldClass}
            type="number"
            min="0"
            step="0.01"
            placeholder="0"
            value={start}
            aria-invalid={
              !validation.valid && validation.error.startsWith("Start")
            }
            onChange={(event) => setStart(event.currentTarget.value)}
          />
        </Field>
        <Field label={text("End (s)", "结束（秒）")}>
          <input
            className={fieldClass}
            type="number"
            min="0"
            step="0.01"
            placeholder={text("End", "结束")}
            value={end}
            aria-invalid={
              !validation.valid && validation.error.startsWith("End")
            }
            onChange={(event) => setEnd(event.currentTarget.value)}
          />
        </Field>
      </div>
      <label className="flex min-h-7 items-center gap-2 text-xs text-foreground">
        <input
          className="size-3.5 accent-primary"
          type="checkbox"
          checked={csvHeader}
          disabled={format !== "csv"}
          onChange={(event) => setCsvHeader(event.currentTarget.checked)}
        />
        {text("Include CSV comments and column header", "包含 CSV 注释和列标题")}
      </label>
      <p className="text-[10px] leading-[1.45] text-muted-foreground">
        {text(
          "Export FPS resamples the finished trajectory without running IK again.",
          "导出 FPS 会对完成的轨迹重新采样，不会再次运行 IK。",
        )}
        {hasScene
          ? text(
              " Terrain or object results download as a ZIP bundle.",
              " 含地形或物体的结果将下载为 ZIP 包。",
            )
          : ""}
      </p>
      {!validation.valid && (
        <p className="text-[11px] text-danger" role="alert">
          {localizedValidationError(validation.error, text)}
        </p>
      )}
      {exportUrl ? (
        <a
          className="inline-flex min-h-[30px] items-center justify-center rounded-md border border-primary bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-accent-foreground"
          href={exportUrl}
          download
        >
          {text("Download result", "下载结果")}
        </a>
      ) : (
        <Button size="sm" variant="primary" disabled>
          {text("Download result", "下载结果")}
        </Button>
      )}
    </section>
  );
}
