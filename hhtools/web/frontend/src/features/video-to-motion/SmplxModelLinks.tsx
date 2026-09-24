import {
  isSmplxNeutralMissing,
  SMPLX_DOWNLOAD_URL,
  type GvhmrRuntimeStatus,
} from "./api";
import { useLocaleText } from "@/LocaleProvider";

export function SmplxModelLinks({
  runtime,
}: {
  readonly runtime: GvhmrRuntimeStatus | null;
}) {
  const text = useLocaleText();
  if (!isSmplxNeutralMissing(runtime)) return null;

  return (
    <aside
      className="rounded-md border border-warning-border bg-warning-muted px-2.5 py-2 text-[11px] font-semibold leading-relaxed"
      aria-label={text("SMPL-X model setup", "SMPL-X 模型配置")}
    >
      <a
        className="text-primary hover:underline"
        href={SMPLX_DOWNLOAD_URL}
        target="_blank"
        rel="noreferrer"
      >
        {text("Download SMPL-X model", "下载 SMPL-X 模型")}
      </a>
    </aside>
  );
}
