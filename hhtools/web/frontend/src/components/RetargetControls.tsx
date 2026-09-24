import { Field, fieldClass } from "@/components/Field";
import { Button } from "@/components/ui/button";
import { useLocaleText } from "@/LocaleProvider";

export function RetargetControls({
  fpsPlaceholder,
  disabledReason,
}: {
  fpsPlaceholder: string;
  disabledReason: string;
}) {
  const text = useLocaleText();
  return (
    <div className="grid gap-2.5">
      <div className="grid grid-cols-2 gap-2">
        <Field label={text("Solver", "求解器")}>
          <select className={fieldClass} defaultValue="newton" disabled>
            <option value="newton">Newton IK</option>
            <option value="interaction-mesh">Interaction-Mesh</option>
          </select>
        </Field>
        <Field label={text("Retarget FPS", "重定向 FPS")}>
          <input className={fieldClass} placeholder={fpsPlaceholder} disabled />
        </Field>
      </div>
      <Button variant="primary" size="sm" disabled>
        {text("Start Retarget", "开始重定向")}
      </Button>
      <p className="text-xs leading-[1.4] text-muted-foreground">
        {disabledReason}
      </p>
    </div>
  );
}
