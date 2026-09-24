import type { StageVec3 } from "./types";

export type CalibrationAngleUnit = "rad" | "deg";

export interface CalibrationJointWorld {
  readonly pivot: StageVec3;
  readonly axis: StageVec3;
}

export interface CalibrationInteractionJoint {
  readonly name: string;
  readonly child_link?: string;
  readonly lower?: number;
  readonly upper?: number;
  readonly type?: string;
}

/** Controlled calibration state published by H2R or R2R to the shared Stage. */
export interface CalibrationInteractionModel {
  readonly jointQ: Readonly<Record<string, number>>;
  readonly jointLimits: readonly CalibrationInteractionJoint[];
  readonly jointWorld: Readonly<Record<string, CalibrationJointWorld>>;
  readonly groundOffsetZ: number;
  readonly angleUnit: CalibrationAngleUnit;
  readonly selectedJoint: string | null;
  readonly disabled?: boolean;
  readonly onJointChange: (name: string, value: number) => void;
  readonly onSelectedJointChange: (name: string | null) => void;
  readonly onAngleUnitChange: (unit: CalibrationAngleUnit) => void;
}
