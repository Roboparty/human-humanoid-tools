import { fieldClass } from "@/components/Field";
import { Button } from "@/components/ui/button";
import { useLocaleText } from "@/LocaleProvider";

export function RobotPicker({
  label,
  status,
}: {
  label: string;
  status: string;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2.5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
        <select
          className={fieldClass}
          aria-label={label}
          defaultValue=""
          disabled
        >
          <option value="">{text("No robots available", "没有可用机器人")}</option>
        </select>
        <Button size="sm" disabled>
          {text("Import robot", "导入机器人")}
        </Button>
      </div>
      <Button variant="primary" size="sm" disabled>
        {text("Load robot", "加载机器人")}
      </Button>
      <p className="text-xs text-muted-foreground">{status}</p>
    </div>
  );
}
