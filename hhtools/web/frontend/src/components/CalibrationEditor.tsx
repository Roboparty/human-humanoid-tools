import { useEffect, useMemo, useRef, useState } from "react";

import { useLocaleText } from "@/LocaleProvider";
import type { CalibrationDisplayOptions } from "@/stage/calibrationDisplay";
import { updateCalibrationDisplay } from "@/stage/calibrationDisplay";
import { prepareReferenceSkeleton } from "@/stage/referenceSkeleton";
import type { StageMotionPayload, StageRobotPayload } from "@/stage/types";

import { ValidationSummary } from "./ValidationSummary";
import { calibrationValidationFacts } from "./validationFacts";
import { Button } from "./ui/button";
import {
  angleForDisplay,
  angleFromDisplay,
  calibrationJointMatches,
  clampCalibrationValue,
  formatCalibrationAngle,
  isNearCalibrationLimit,
  normalizeCalibrationValues,
  resolveCalibrationJointLimits,
  setCalibrationJointValue,
  zeroCalibrationRegion,
  zeroCalibrationValues,
  type CalibrationAngleUnit,
  type CalibrationJointLimit,
  type CalibrationJointRegion,
  type ResolvedCalibrationJointLimit,
} from "./calibrationEditorState";

interface CalibrationEditorProps {
  readonly limits: readonly CalibrationJointLimit[];
  readonly value: Readonly<Record<string, number>>;
  readonly baseline: Readonly<Record<string, number>>;
  readonly hasSavedBaseline: boolean;
  readonly reference: StageMotionPayload;
  readonly robot: StageRobotPayload;
  readonly display: CalibrationDisplayOptions;
  readonly angleUnit?: CalibrationAngleUnit;
  readonly selectedJoint?: string | null;
  readonly disabled?: boolean;
  readonly saving?: boolean;
  readonly suggesting?: boolean;
  readonly assistantValidation?: {
    readonly valid: boolean;
    readonly score: number;
    readonly alignment_errors: readonly string[];
    readonly alignment_warnings: readonly string[];
  } | null;
  readonly onChange: (value: Record<string, number>) => void;
  readonly onDisplayChange: (value: CalibrationDisplayOptions) => void;
  readonly onAngleUnitChange?: (unit: CalibrationAngleUnit) => void;
  readonly onJointSelected?: (name: string) => void;
  readonly onCancel: () => void;
  readonly onSuggest?: () => void;
  readonly onSave: () => void;
}

const numberClass =
  "h-7 w-[76px] rounded-md border border-border bg-surface px-2 text-right text-[11px] tabular-nums text-foreground outline-none focus:border-primary disabled:cursor-not-allowed disabled:opacity-50";

type RegionFilter = CalibrationJointRegion | "all";
type ComparisonMode = "current" | "saved" | "zero";

const regions: readonly {
  value: RegionFilter;
  english: string;
  chinese: string;
}[] = [
  { value: "all", english: "All", chinese: "全部" },
  { value: "torso", english: "Torso", chinese: "躯干" },
  { value: "left-arm", english: "L arm", chinese: "左臂" },
  { value: "right-arm", english: "R arm", chinese: "右臂" },
  { value: "left-leg", english: "L leg", chinese: "左腿" },
  { value: "right-leg", english: "R leg", chinese: "右腿" },
  { value: "head", english: "Head", chinese: "头部" },
  { value: "hands", english: "Hands", chinese: "手部" },
];

function CalibrationJointRow({
  limit,
  value,
  unit,
  selected,
  disabled,
  onSelect,
  onChange,
}: {
  readonly limit: ResolvedCalibrationJointLimit;
  readonly value: number;
  readonly unit: CalibrationAngleUnit;
  readonly selected: boolean;
  readonly disabled: boolean;
  readonly onSelect: () => void;
  readonly onChange: (valueRad: number) => void;
}) {
  const text = useLocaleText();
  const editing = useRef(false);
  const [numberValue, setNumberValue] = useState(() =>
    limit.type === "prismatic" ? value.toFixed(3) : formatCalibrationAngle(value, unit),
  );
  const nearLimit = isNearCalibrationLimit(value, limit);
  const linear = limit.type === "prismatic";
  const displayValue = (next: number) =>
    linear ? next : angleForDisplay(next, unit);
  const storedValue = (next: number) =>
    linear ? next : angleFromDisplay(next, unit);
  const formatValue = (next: number) =>
    linear ? next.toFixed(3) : formatCalibrationAngle(next, unit);

  useEffect(() => {
    if (!editing.current) setNumberValue(formatValue(value));
  }, [linear, unit, value]);

  const commit = (raw: string) => {
    const parsed = raw.trim() ? Number(raw) : Number.NaN;
    if (!Number.isFinite(parsed)) {
      setNumberValue(formatValue(value));
      return;
    }
    const next = clampCalibrationValue(storedValue(parsed), limit);
    onChange(next);
    setNumberValue(formatValue(next));
  };

  return (
    <div
      className={`grid grid-cols-[minmax(0,1fr)_minmax(72px,1fr)_76px] items-center gap-2 rounded-sm px-1 py-0.5 text-[11px] ${selected ? "bg-accent" : ""}`}
      onPointerDown={onSelect}
    >
      <span
        className={nearLimit ? "truncate text-warning" : "truncate text-muted-foreground"}
        title={limit.name}
      >
        {limit.name}
      </span>
      <input
        type="range"
        min={limit.lower}
        max={limit.upper}
        step="0.001"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        className="h-4 min-w-0 accent-primary"
        aria-label={text(`${limit.name} angle`, `${limit.name} 角度`)}
      />
      <input
        type="number"
        min={displayValue(limit.lower)}
        max={displayValue(limit.upper)}
        step={!linear && unit === "deg" ? "0.1" : "0.001"}
        value={numberValue}
        disabled={disabled}
        onFocus={() => {
          editing.current = true;
        }}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          setNumberValue(raw);
          const parsed = raw.trim() ? Number(raw) : Number.NaN;
          if (Number.isFinite(parsed)) onChange(storedValue(parsed));
        }}
        onBlur={(event) => {
          editing.current = false;
          commit(event.currentTarget.value);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        className={`${numberClass} ${nearLimit ? "border-warning/55" : ""}`}
        aria-label={`${limit.name} ${
          linear
            ? text("metres", "米")
            : unit === "deg"
              ? text("degrees", "度")
              : text("radians", "弧度")
        }`}
        title={
          linear
            ? text("Translation in metres", "平移量，单位为米")
            : unit === "deg"
              ? text("Degrees; stored in radians", "显示为度，内部以弧度存储")
              : text("Angle in radians", "角度，单位为弧度")
        }
      />
    </div>
  );
}

/** Shared controlled editor for H2R and R2R calibration sessions. */
export function CalibrationEditor({
  limits,
  value,
  baseline,
  hasSavedBaseline,
  reference,
  robot,
  display,
  angleUnit: controlledAngleUnit,
  selectedJoint = null,
  disabled = false,
  saving = false,
  suggesting = false,
  assistantValidation = null,
  onChange,
  onDisplayChange,
  onAngleUnitChange,
  onJointSelected,
  onCancel,
  onSuggest,
  onSave,
}: CalibrationEditorProps) {
  const text = useLocaleText();
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState<RegionFilter>("all");
  const [localAngleUnit, setLocalAngleUnit] = useState<CalibrationAngleUnit>("rad");
  const unit = controlledAngleUnit ?? localAngleUnit;
  const publishAngleUnit = onAngleUnitChange ?? setLocalAngleUnit;
  const [comparison, setComparison] = useState<ComparisonMode>("current");
  const currentDraft = useRef(normalizeCalibrationValues(limits, value));
  const resolved = useMemo(
    () => resolveCalibrationJointLimits(limits, value),
    [limits, value],
  );
  const visibleLimits = useMemo(
    () => resolved.filter((limit) => calibrationJointMatches(limit.name, query, region)),
    [query, region, resolved],
  );
  const mappedLandmarks = useMemo(
    () => prepareReferenceSkeleton(reference, robot).mappings.length,
    [reference, robot],
  );
  const updateDisplay = (patch: Partial<CalibrationDisplayOptions>) => {
    onDisplayChange(updateCalibrationDisplay(display, patch));
  };
  const publishEdit = (next: Readonly<Record<string, number>>) => {
    const normalized = normalizeCalibrationValues(limits, next);
    currentDraft.current = normalized;
    setComparison("current");
    onChange(normalized);
  };
  const showComparison = (next: ComparisonMode) => {
    if (comparison === "current") {
      currentDraft.current = normalizeCalibrationValues(limits, value);
    }
    const target =
      next === "zero"
        ? zeroCalibrationValues(limits, value)
        : next === "saved"
          ? normalizeCalibrationValues(limits, baseline)
          : currentDraft.current;
    setComparison(next);
    onChange(target);
  };

  return (
    <div className="grid gap-2.5 rounded-md border border-border-subtle bg-background p-2.5">
      <div className="grid gap-1.5 border-b border-border-subtle pb-2.5 text-[11px]">
        <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-1.5">
          <input
            type="search"
            value={query}
            placeholder={text("Search joints", "搜索关节")}
            autoComplete="off"
            disabled={disabled}
            onChange={(event) => setQuery(event.currentTarget.value)}
            className="h-7 min-w-0 rounded-md border border-border bg-surface px-2 text-[11px] text-foreground outline-none focus:border-primary"
            aria-label={text("Search calibration joints", "搜索标定关节")}
          />
          <span className="min-w-9 text-center tabular-nums text-muted-foreground">
            {visibleLimits.length}/{resolved.length}
          </span>
          <div className="grid grid-cols-2 overflow-hidden rounded-md border border-border-subtle">
            {(["rad", "deg"] as const).map((option) => (
              <button
                key={option}
                type="button"
                disabled={disabled}
                aria-pressed={unit === option}
                onClick={() => publishAngleUnit(option)}
                className={`min-h-7 px-2 text-[10px] font-semibold ${
                  unit === option
                    ? "bg-accent text-accent-foreground"
                    : "bg-surface text-muted-foreground hover:bg-background"
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
        <div
          className="flex flex-wrap gap-1"
          role="group"
          aria-label={text("Joint regions", "关节区域")}
        >
          {regions.map((option) => (
            <button
              key={option.value}
              type="button"
              disabled={disabled}
              aria-pressed={region === option.value}
              onClick={() => setRegion(option.value)}
              className={`min-h-6 rounded-md border px-2 text-[10px] font-semibold ${
                region === option.value
                  ? "border-primary bg-accent text-accent-foreground"
                  : "border-border-subtle bg-surface text-muted-foreground hover:border-border"
              }`}
            >
              {text(option.english, option.chinese)}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-1.5">
          <div
            className="grid grid-cols-3 overflow-hidden rounded-md border border-border-subtle"
            role="group"
            aria-label={text("Pose comparison", "姿势对比")}
          >
            {([
              ["current", text("Current", "当前")],
              ["saved", text("Saved", "已保存")],
              ["zero", text("URDF zero", "URDF 零位")],
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                disabled={disabled || (mode === "saved" && !hasSavedBaseline)}
                aria-pressed={comparison === mode}
                onClick={() => showComparison(mode)}
                className={`min-h-7 min-w-0 truncate px-1.5 text-[10px] font-semibold ${
                  comparison === mode
                    ? "bg-accent text-accent-foreground"
                    : "bg-surface text-muted-foreground hover:bg-background disabled:opacity-40"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <Button
            size="sm"
            disabled={disabled}
            className="min-h-7 px-2 text-[10px]"
            onClick={() =>
              publishEdit(zeroCalibrationRegion(limits, value, region))
            }
          >
            {text("Zero region", "区域归零")}
          </Button>
        </div>
      </div>

      <div className="grid gap-2 border-b border-border-subtle pb-2.5 text-[11px] text-foreground">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="font-semibold text-muted-foreground">
            {text("Stage display", "舞台显示")}
          </span>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={display.mappedOnly}
              disabled={disabled}
              onChange={(event) => updateDisplay({ mappedOnly: event.currentTarget.checked })}
              className="size-3.5 accent-primary"
            />
            {text("Mapped only", "仅已映射")}
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={display.labels}
              disabled={disabled}
              onChange={(event) => updateDisplay({ labels: event.currentTarget.checked })}
              className="size-3.5 accent-primary"
            />
            {text("Labels", "标签")}
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={display.mappingLines}
              disabled={disabled}
              onChange={(event) => updateDisplay({ mappingLines: event.currentTarget.checked })}
              className="size-3.5 accent-primary"
            />
            {text("Link lines", "连接线")}
          </label>
          <span className="ml-auto text-muted-foreground">
            {text(`${mappedLandmarks} mapped`, `已映射 ${mappedLandmarks} 个`)}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <label className="grid gap-1">
            <span className="flex justify-between gap-2">
              {text("Reference", "参考姿势")}
              <span className="tabular-nums text-muted-foreground">
                {Math.round(display.referenceOpacity * 100)}%
              </span>
            </span>
            <input
              type="range"
              min="0.15"
              max="1"
              step="0.05"
              value={display.referenceOpacity}
              disabled={disabled}
              onChange={(event) =>
                updateDisplay({ referenceOpacity: Number(event.currentTarget.value) })
              }
              className="h-4 w-full accent-primary"
            />
          </label>
          <label className="grid gap-1">
            <span className="flex justify-between gap-2">
              {text("Robot", "机器人")}
              <span className="tabular-nums text-muted-foreground">
                {Math.round(display.robotOpacity * 100)}%
              </span>
            </span>
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.05"
              value={display.robotOpacity}
              disabled={disabled}
              onChange={(event) =>
                updateDisplay({ robotOpacity: Number(event.currentTarget.value) })
              }
              className="h-4 w-full accent-primary"
            />
          </label>
        </div>
      </div>

      <ValidationSummary
        items={calibrationValidationFacts(robot, limits, value, text)}
        label={text("Calibration validation", "标定验证")}
      />

      {onSuggest ? (
        <Button
          size="sm"
          disabled={disabled}
          onClick={onSuggest}
          title={text(
            "Generate a limit-constrained pose proposal; it will not be saved automatically here.",
            "生成满足关节限位的姿态建议；此处不会自动保存。",
          )}
        >
          {suggesting
            ? text("Proposing…", "正在生成建议…")
            : text("Auto-propose pose", "自动建议姿态")}
        </Button>
      ) : null}
      {assistantValidation ? (
        <p
          className={`rounded-md border px-2.5 py-2 text-[11px] ${
            assistantValidation.valid
              ? "border-success/35 bg-success-muted text-success"
              : "border-warning/40 bg-warning-muted text-warning"
          }`}
          role="status"
        >
          {assistantValidation.valid
            ? text(
                `Automatic checks passed · score ${assistantValidation.score.toFixed(2)}.`,
                `自动检查已通过 · 评分 ${assistantValidation.score.toFixed(2)}。`,
              )
            : text(
                `Needs adjustment · ${assistantValidation.alignment_errors.length} alignment errors.`,
                `仍需调整 · ${assistantValidation.alignment_errors.length} 个对齐错误。`,
              )}
          {assistantValidation.alignment_warnings.length
            ? text(
                ` ${assistantValidation.alignment_warnings.length} warnings remain.`,
                ` 仍有 ${assistantValidation.alignment_warnings.length} 项警告。`,
              )
            : ""}
        </p>
      ) : null}

      <div className="grid max-h-64 gap-1.5 overflow-y-auto pr-1">
        {visibleLimits.map((limit) => (
          <CalibrationJointRow
            key={limit.name}
            limit={limit}
            value={value[limit.name] ?? 0}
            unit={unit}
            selected={selectedJoint === limit.name}
            disabled={disabled}
            onSelect={() => onJointSelected?.(limit.name)}
            onChange={(next) =>
              publishEdit(
                setCalibrationJointValue(limits, value, limit.name, next),
              )
            }
          />
        ))}
        {visibleLimits.length === 0 && (
          <p className="py-2 text-center text-[11px] text-muted-foreground">
            {text("No matching joints", "没有匹配的关节")}
          </p>
        )}
      </div>

      <div className="grid grid-cols-4 gap-1.5 border-t border-border-subtle pt-2.5 max-[420px]:grid-cols-2">
        <Button
          size="sm"
          disabled={disabled}
          onClick={() => showComparison("zero")}
        >
          {text("Zero", "归零")}
        </Button>
        <Button
          size="sm"
          disabled={disabled || !hasSavedBaseline}
          title={
            hasSavedBaseline
              ? text("Restore the saved calibration", "恢复已保存的标定")
              : text("No saved calibration", "没有已保存的标定")
          }
          onClick={() => showComparison("saved")}
        >
          {text("Reset", "重置")}
        </Button>
        <Button size="sm" disabled={disabled} onClick={onCancel}>
          {text("Cancel", "取消")}
        </Button>
        <Button variant="primary" size="sm" disabled={disabled} onClick={onSave}>
          {saving ? text("Saving…", "保存中…") : text("Save", "保存")}
        </Button>
      </div>
    </div>
  );
}
