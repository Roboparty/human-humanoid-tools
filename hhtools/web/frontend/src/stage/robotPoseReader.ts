import type * as THREE from "three";

/** Minimal read-only bridge from RobotLayer to calibration overlays. */
export interface RobotLinkPoseReader {
  getLinkWorldPosition(link: string, output: THREE.Vector3): boolean;
  getLinkWorldQuaternion(link: string, output: THREE.Quaternion): boolean;
}

