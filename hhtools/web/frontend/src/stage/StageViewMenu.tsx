import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { useLocaleText } from "@/LocaleProvider";
import { cn } from "@/lib/utils";
import {
  applyStageMenuValue,
  projectStageMenuValue,
  type StandardStageLayerAvailability,
  type StandardStageLayerId,
} from "./presentation";
import type {
  R2rLayerAvailability,
  R2rStageLayerId,
  StageLayerId,
} from "./types";

type LocalizedText = readonly [english: string, chinese: string];

interface StageLayer {
  readonly id: StageLayerId;
  readonly legacyId: string;
  readonly label: LocalizedText;
  readonly accessibleLabel?: LocalizedText;
  readonly title: LocalizedText;
  readonly family:
    | "skeleton"
    | "body"
    | "scene"
    | "scaled-skeleton"
    | "scaled-scene"
    | "robot";
}

interface StageLayerRow {
  readonly id: string;
  readonly label?: LocalizedText;
  readonly layers: readonly StageLayer[];
}

const layerRows: readonly StageLayerRow[] = [
  {
    id: "motion",
    layers: [
      {
        id: "skeleton",
        legacyId: "tg-skeleton",
        label: ["Skeleton", "骨架"],
        title: ["Skeleton", "骨架"],
        family: "skeleton",
      },
      {
        id: "body",
        legacyId: "tg-mesh",
        label: ["Body", "人体"],
        title: ["Body", "人体"],
        family: "body",
      },
      {
        id: "objects",
        legacyId: "tg-env",
        label: ["Objects/Terrain", "物体/地形"],
        title: ["Objects/Terrain", "物体/地形"],
        family: "scene",
      },
    ],
  },
  {
    id: "robot",
    layers: [
      {
        id: "scaled-skeleton",
        legacyId: "tg-scaled",
        label: ["Scaled", "缩放后"],
        accessibleLabel: ["Scaled Skeleton", "缩放后骨架"],
        title: ["Scaled Skeleton", "缩放后骨架"],
        family: "scaled-skeleton",
      },
      {
        id: "scaled-scene",
        legacyId: "tg-scaled-env",
        label: ["Scaled", "缩放后"],
        accessibleLabel: ["Scaled Scene", "缩放后场景"],
        title: ["Scaled Scene", "缩放后场景"],
        family: "scaled-scene",
      },
      {
        id: "robot",
        legacyId: "tg-robot",
        label: ["Robot", "机器人"],
        title: ["Robot", "机器人"],
        family: "robot",
      },
    ],
  },
];

const r2rLayerRows: readonly StageLayerRow[] = [
  {
    id: "r2r-src",
    label: ["Source", "源"],
    layers: [
      {
        id: "r2r-source-robot",
        legacyId: "r2r-tg-src-robot",
        label: ["Robot", "机器人"],
        accessibleLabel: ["Source Robot", "源机器人"],
        title: ["Robot", "机器人"],
        family: "robot",
      },
      {
        id: "r2r-source-skeleton",
        legacyId: "r2r-tg-src-skel",
        label: ["Skeleton", "骨架"],
        accessibleLabel: ["Source Skeleton", "源骨架"],
        title: ["Skeleton", "骨架"],
        family: "skeleton",
      },
      {
        id: "r2r-source-scene",
        legacyId: "r2r-tg-src-env",
        label: ["Objects/Terrain", "物体/地形"],
        accessibleLabel: ["Source Objects/Terrain", "源物体/地形"],
        title: ["Objects/Terrain", "物体/地形"],
        family: "scene",
      },
    ],
  },
  {
    id: "r2r-tgt",
    label: ["Target", "目标"],
    layers: [
      {
        id: "r2r-target-robot",
        legacyId: "r2r-tg-tgt-robot",
        label: ["Robot", "机器人"],
        accessibleLabel: ["Target Robot", "目标机器人"],
        title: ["Robot", "机器人"],
        family: "robot",
      },
      {
        id: "r2r-target-skeleton",
        legacyId: "r2r-tg-tgt-skel",
        label: ["Skeleton", "骨架"],
        accessibleLabel: ["Target Skeleton", "目标骨架"],
        title: ["Skeleton", "骨架"],
        family: "scaled-skeleton",
      },
      {
        id: "r2r-target-scene",
        legacyId: "r2r-tg-tgt-env",
        label: ["Objects/Terrain", "物体/地形"],
        accessibleLabel: ["Target Objects/Terrain", "目标物体/地形"],
        title: ["Objects/Terrain", "物体/地形"],
        family: "scaled-scene",
      },
    ],
  },
];

const activeFamilyClass: Record<StageLayer["family"], string> = {
  skeleton:
    "data-[state=on]:bg-stage-skeleton data-[state=on]:text-white data-[state=on]:hover:bg-stage-skeleton",
  body:
    "data-[state=on]:bg-stage-body data-[state=on]:text-white data-[state=on]:hover:bg-stage-body",
  scene:
    "data-[state=on]:bg-stage-scene data-[state=on]:text-white data-[state=on]:hover:bg-stage-scene",
  "scaled-skeleton":
    "data-[state=on]:bg-stage-scaled-skeleton data-[state=on]:text-[#211600] data-[state=on]:hover:bg-stage-scaled-skeleton",
  "scaled-scene":
    "data-[state=on]:bg-stage-scaled-scene data-[state=on]:text-white data-[state=on]:hover:bg-stage-scaled-scene",
  robot:
    "data-[state=on]:bg-stage-robot data-[state=on]:text-white data-[state=on]:hover:bg-stage-robot",
};

interface StageViewMenuProps {
  value: StageLayerId[];
  onValueChange(value: StageLayerId[]): void;
  availability: StandardStageLayerAvailability;
  calibration?: boolean;
  r2rAvailability?: R2rLayerAvailability;
}

export function StageViewMenu({
  value,
  onValueChange,
  availability,
  calibration = false,
  r2rAvailability,
}: StageViewMenuProps) {
  const text = useLocaleText();
  const r2r = r2rAvailability !== undefined;
  const rows = r2r ? r2rLayerRows : layerRows;
  const menuValue = projectStageMenuValue(value, calibration, r2r);
  const available = (layer: StageLayer): boolean =>
    layer.id.startsWith("r2r-")
      ? Boolean(r2rAvailability?.[layer.id as R2rStageLayerId])
      : availability[layer.id as StandardStageLayerId];
  const locked = (layer: StageLayer): boolean =>
    calibration && (r2r || layer.id !== "robot");

  return (
    <ToggleGroup
      type="multiple"
      value={menuValue}
      onValueChange={(next) =>
        onValueChange(
          applyStageMenuValue(value, next as StageLayerId[], calibration, r2r),
        )
      }
      size="sm"
      aria-label={text("Stage visibility", "预览图层显示")}
      className="stage-view-menu absolute top-3 left-3 z-[25] flex w-fit max-w-[calc(100%_-_36px)] flex-col items-stretch gap-1 rounded-lg border border-border-subtle bg-surface/[.85] px-2 py-1.5 shadow-[0_1px_3px_rgba(0,0,0,0.08)] backdrop-blur-[20px] @max-[440px]:top-2 @max-[440px]:left-2 @max-[440px]:max-w-[calc(100%_-_16px)]"
    >
      {rows.map((row) => (
        <div
          key={row.id}
          data-row={row.id}
          className="flex flex-nowrap items-center gap-[3px]"
        >
          {row.label && (
            <span className="min-w-[2.2em] shrink-0 text-center text-[11px] font-bold text-muted-foreground/60">
              {text(...row.label)}
            </span>
          )}
          {row.layers.map((layer) => {
            const label = text(...layer.label);
            const accessibleLabel = text(
              ...(layer.accessibleLabel ?? layer.label),
            );
            return (
              <ToggleGroupItem
                key={layer.id}
                id={layer.legacyId}
                value={layer.id}
                disabled={!available(layer) || locked(layer)}
                title={text(...layer.title)}
                aria-label={accessibleLabel}
                data-family={layer.family}
                className={cn(
                  "group h-auto w-auto cursor-pointer gap-[5px] rounded-sm border-0 bg-transparent px-3 py-1.5 text-xs leading-[normal] font-semibold text-muted-foreground transition-[background-color,color,opacity] duration-150 hover:bg-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-ring disabled:pointer-events-auto disabled:cursor-not-allowed disabled:grayscale-[.35] disabled:data-[state=off]:opacity-55 disabled:data-[state=on]:opacity-100 @max-[440px]:gap-1 @max-[440px]:px-1",
                  activeFamilyClass[layer.family],
                )}
              >
                <span
                  className="stage-layer-eye size-[13px] shrink-0 bg-current opacity-55 grayscale-[.6] [mask:url(/icons/stage/eye.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/stage/eye.svg)_center/contain_no-repeat] group-hover:opacity-80 group-data-[state=on]:opacity-100 group-data-[state=on]:grayscale-0"
                  aria-hidden="true"
                />
                <span>{label}</span>
              </ToggleGroupItem>
            );
          })}
        </div>
      ))}
    </ToggleGroup>
  );
}
