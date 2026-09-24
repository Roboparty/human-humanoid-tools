import { Button } from "@/components/ui/button";
import { useLocaleText } from "@/LocaleProvider";

export type ImportAssetKind = "motion" | "robot";

const labels: Readonly<Record<ImportAssetKind, readonly [string, string]>> = {
  motion: ["Import motion", "导入动作"],
  robot: ["Import robot", "导入机器人"],
};

/** Opens the owning asset workspace; importing remains the asset view's job. */
export function AssetImportButton({
  kind,
  onClick,
}: {
  readonly kind: ImportAssetKind;
  readonly onClick: () => void;
}) {
  const text = useLocaleText();
  const label = text(...labels[kind]);
  return (
    <Button
      size="sm"
      variant="primaryOutline"
      className="shrink-0 gap-1.5 px-2.5"
      aria-label={label}
      title={text(
        `Open the ${kind === "motion" ? "Motion" : "Robot"} workspace`,
        `打开${kind === "motion" ? "动作" : "机器人"}工作区`,
      )}
      onClick={onClick}
    >
      <span
        className="size-3.5 shrink-0 bg-current [mask:url(/icons/common/upload.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/upload.svg)_center/contain_no-repeat]"
        aria-hidden="true"
      />
      <span>{label}</span>
    </Button>
  );
}
