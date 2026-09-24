import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";

import {
  calibrationMappingKey,
  calibrationMappingLabel,
  layoutCalibrationLabels,
  type CalibrationMappingProjection,
} from "./calibrationOverlay";
import type { ReferenceJointMapping } from "./referenceSkeleton";

export interface CalibrationMappingOverlayHandle {
  clear(): void;
  present(projections: readonly CalibrationMappingProjection[]): void;
}

interface CalibrationMappingOverlayProps {
  readonly mappings: readonly ReferenceJointMapping[];
  readonly visible: boolean;
  readonly labels?: boolean;
  readonly mappingLines?: boolean;
}

/** React owns the nodes; the renderer only updates their projected coordinates. */
export const CalibrationMappingOverlay = forwardRef<
  CalibrationMappingOverlayHandle,
  CalibrationMappingOverlayProps
>(function CalibrationMappingOverlay(
  { mappings, visible, labels: showLabels = true, mappingLines: showLines = true },
  forwardedRef,
) {
  const labels = useRef(new Map<string, HTMLSpanElement>());
  const lines = useRef(new Map<string, SVGLineElement>());

  const clear = () => {
    for (const label of labels.current.values()) label.style.display = "none";
    for (const line of lines.current.values()) line.style.display = "none";
  };

  useImperativeHandle(
    forwardedRef,
    () => ({
      clear,
      present(projections) {
        clear();
        for (const projection of projections) {
          const label = labels.current.get(projection.key);
          const line = lines.current.get(projection.key);
          if (!label || !line) continue;
          label.style.display = showLabels ? "block" : "none";
          line.style.display = showLines ? "inline" : "none";
          line.setAttribute("x1", String(projection.referenceX));
          line.setAttribute("y1", String(projection.referenceY));
          line.setAttribute("x2", String(projection.targetX));
          line.setAttribute("y2", String(projection.targetY));
        }
        if (showLabels) {
          const firstLabel = labels.current.values().next().value;
          const host = firstLabel?.parentElement;
          if (!host) return;
          const layout = layoutCalibrationLabels(
            projections.flatMap((projection) => {
              const label = labels.current.get(projection.key);
              return label
                ? [{
                    key: projection.key,
                    anchorX: projection.referenceX,
                    anchorY: projection.referenceY,
                    width: label.offsetWidth,
                    height: label.offsetHeight,
                  }]
                : [];
            }),
            host.clientWidth,
            host.clientHeight,
          );
          for (const position of layout) {
            const label = labels.current.get(position.key);
            if (!label) continue;
            label.style.left = `${position.left}px`;
            label.style.top = `${position.top}px`;
          }
        }
      },
    }),
  );

  useEffect(() => {
    clear();
  }, [mappings, visible]);

  return (
    <>
      <svg
        className="pointer-events-none absolute inset-0 z-[9] size-full overflow-hidden"
        aria-hidden="true"
        style={{ display: visible && showLines ? undefined : "none" }}
      >
        <g>
          {mappings.map((mapping) => {
            const key = calibrationMappingKey(mapping);
            return (
              <line
                key={key}
                ref={(element) => {
                  if (element) lines.current.set(key, element);
                  else lines.current.delete(key);
                }}
                style={{
                  display: "none",
                  stroke: "color-mix(in srgb, var(--accent) 62%, transparent)",
                  strokeWidth: 1,
                  strokeDasharray: "4 4",
                  vectorEffect: "non-scaling-stroke",
                }}
              />
            );
          })}
        </g>
      </svg>
      <div
        className="pointer-events-none absolute inset-0 z-[9] size-full overflow-hidden"
        aria-hidden="true"
        hidden={!visible || !showLabels}
      >
        {mappings.map((mapping) => {
          const key = calibrationMappingKey(mapping);
          const label = calibrationMappingLabel(mapping);
          const separator = label.lastIndexOf(" · ");
          return (
            <span
              key={key}
              ref={(element) => {
                if (element) labels.current.set(key, element);
                else labels.current.delete(key);
              }}
              className="absolute max-w-40 whitespace-nowrap rounded-[4px] border px-1.5 py-[3px]"
              style={{
                display: "none",
                borderColor: "color-mix(in srgb, var(--accent) 30%, transparent)",
                background: "color-mix(in srgb, var(--surface) 90%, transparent)",
                color: "var(--text-muted)",
                boxShadow: "0 1px 3px rgba(0, 0, 0, 0.08)",
                font: "600 9px/1.2 -apple-system, sans-serif",
              }}
            >
              <strong className="font-bold text-primary">
                {label.slice(0, separator)}
              </strong>
              {label.slice(separator)}
            </span>
          );
        })}
      </div>
    </>
  );
});
