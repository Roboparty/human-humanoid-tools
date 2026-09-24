export interface CalibrationDisplayOptions {
  readonly mappedOnly: boolean;
  readonly labels: boolean;
  readonly mappingLines: boolean;
  readonly referenceOpacity: number;
  readonly robotOpacity: number;
}

export const DEFAULT_CALIBRATION_DISPLAY: CalibrationDisplayOptions = {
  mappedOnly: true,
  labels: true,
  mappingLines: true,
  referenceOpacity: 0.82,
  robotOpacity: 0.72,
};

function clamp(value: number, lower: number, upper: number): number {
  return Math.min(upper, Math.max(lower, value));
}

export function updateCalibrationDisplay(
  current: CalibrationDisplayOptions,
  patch: Partial<CalibrationDisplayOptions>,
): CalibrationDisplayOptions {
  return {
    ...current,
    ...patch,
    referenceOpacity: clamp(
      patch.referenceOpacity ?? current.referenceOpacity,
      0.15,
      1,
    ),
    robotOpacity: clamp(patch.robotOpacity ?? current.robotOpacity, 0.2, 1),
  };
}

