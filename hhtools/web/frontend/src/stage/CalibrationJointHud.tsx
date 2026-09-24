import { useEffect, useState } from "react";

import {
  angleForDisplay,
  angleFromDisplay,
  formatCalibrationAngle,
} from "@/components/calibrationEditorState";

import type { CalibrationInteractionModel } from "./calibrationInteraction";
import {
  calibrationDragValue,
  resolvedCalibrationDragLimit,
} from "./calibrationManipulatorMath";

function formatHudValue(
  value: number,
  linear: boolean,
  unit: CalibrationInteractionModel["angleUnit"],
): string {
  return linear ? value.toFixed(3) : formatCalibrationAngle(value, unit);
}

function hudValue(
  value: number,
  linear: boolean,
  unit: CalibrationInteractionModel["angleUnit"],
): number {
  return linear ? value : angleForDisplay(value, unit);
}

function storedValue(
  value: number,
  linear: boolean,
  unit: CalibrationInteractionModel["angleUnit"],
): number {
  return linear ? value : angleFromDisplay(value, unit);
}

export function CalibrationJointHud({
  interaction,
  jointName,
  onClose,
}: {
  readonly interaction: CalibrationInteractionModel;
  readonly jointName: string;
  readonly onClose: () => void;
}) {
  const joint = interaction.jointLimits.find((item) => item.name === jointName);
  const linear = joint?.type === "prismatic";
  const limit = resolvedCalibrationDragLimit(joint ?? {});
  const value = calibrationDragValue(
    interaction.jointQ[jointName] ?? 0,
    0,
    limit,
  );
  const [draft, setDraft] = useState(() =>
    formatHudValue(value, linear, interaction.angleUnit),
  );

  useEffect(() => {
    setDraft(formatHudValue(value, linear, interaction.angleUnit));
  }, [interaction.angleUnit, linear, value]);

  function commit(raw: string): void {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      setDraft(formatHudValue(value, linear, interaction.angleUnit));
      return;
    }
    const next = calibrationDragValue(
      storedValue(parsed, linear, interaction.angleUnit),
      0,
      limit,
    );
    interaction.onJointChange(jointName, next);
  }

  return (
    <section
      className="absolute top-3 right-3 z-30 w-[230px] rounded-md border border-border bg-surface/95 p-2.5 shadow-[0_5px_22px_rgba(15,23,42,0.14)] backdrop-blur-xl"
      aria-label={`Joint ${jointName}`}
    >
      <div className="mb-2 flex items-center gap-2">
        <strong
          className="min-w-0 flex-1 truncate text-xs font-medium text-foreground"
          title={jointName}
        >
          {jointName}
        </strong>
        {linear ? (
          <span className="rounded-md border border-border-subtle px-1.5 py-0.5 text-[10px] text-muted-foreground">
            m
          </span>
        ) : (
          <div className="grid grid-cols-2 overflow-hidden rounded-md border border-border-subtle text-[10px]">
            {(["rad", "deg"] as const).map((unit) => (
              <button
                key={unit}
                type="button"
                className={
                  interaction.angleUnit === unit
                    ? "bg-primary px-1.5 py-0.5 text-primary-foreground"
                    : "px-1.5 py-0.5 text-muted-foreground hover:bg-muted"
                }
                onClick={() => interaction.onAngleUnitChange(unit)}
              >
                {unit}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          className="grid size-5 place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="Close joint controls"
          title="Close"
          onClick={onClose}
        >
          <span
            className="size-3.5 bg-current [mask:url(/icons/common/close.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/common/close.svg)_center/contain_no-repeat]"
            aria-hidden="true"
          />
        </button>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_74px] items-center gap-2">
        <input
          type="range"
          min={limit.lower}
          max={limit.upper}
          step="0.001"
          value={value}
          disabled={interaction.disabled}
          className="h-4 min-w-0 accent-primary"
          aria-label={`${jointName} value`}
          onChange={(event) =>
            interaction.onJointChange(jointName, Number(event.currentTarget.value))
          }
        />
        <input
          type="number"
          min={hudValue(limit.lower, linear, interaction.angleUnit)}
          max={hudValue(limit.upper, linear, interaction.angleUnit)}
          step={!linear && interaction.angleUnit === "deg" ? "0.1" : "0.001"}
          value={draft}
          disabled={interaction.disabled}
          className="h-7 min-w-0 rounded-md border border-border bg-surface px-2 text-right text-[11px] tabular-nums text-foreground outline-none focus:border-primary"
          aria-label={`${jointName} ${linear ? "metres" : interaction.angleUnit}`}
          onChange={(event) => setDraft(event.currentTarget.value)}
          onBlur={(event) => commit(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") event.currentTarget.blur();
          }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[9px] tabular-nums text-muted-foreground">
        <span>{formatHudValue(limit.lower, linear, interaction.angleUnit)}</span>
        <span>{linear ? "m" : interaction.angleUnit}</span>
        <span>{formatHudValue(limit.upper, linear, interaction.angleUnit)}</span>
      </div>
    </section>
  );
}
