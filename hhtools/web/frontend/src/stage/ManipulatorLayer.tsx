import type { ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type {
  CalibrationInteractionJoint,
  CalibrationInteractionModel,
  CalibrationJointWorld,
} from "./calibrationInteraction";
import {
  calibrationDragValue,
  jointAxisParameter,
  jointDragVector,
  jointPlanePoint,
  resolvedCalibrationDragLimit,
  signedJointDragAngle,
} from "./calibrationManipulatorMath";
import type { StageVec3 } from "./types";

interface JointHandle {
  readonly joint: CalibrationInteractionJoint;
  readonly world: CalibrationJointWorld;
}

interface R3fPointerCaptureTarget {
  hasPointerCapture(pointerId: number): boolean;
  setPointerCapture(pointerId: number): void;
  releasePointerCapture(pointerId: number): void;
}

interface RotationDrag {
  readonly kind: "rotation";
  readonly pointerId: number;
  readonly captureTarget: R3fPointerCaptureTarget;
  readonly joint: string;
  readonly pivot: StageVec3;
  readonly axis: StageVec3;
  readonly startVector: StageVec3;
  readonly startValue: number;
  readonly limit: ReturnType<typeof resolvedCalibrationDragLimit>;
}

interface TranslationDrag {
  readonly kind: "translation";
  readonly pointerId: number;
  readonly captureTarget: R3fPointerCaptureTarget;
  readonly joint: string;
  readonly pivot: StageVec3;
  readonly axis: StageVec3;
  readonly startParameter: number;
  readonly startValue: number;
  readonly limit: ReturnType<typeof resolvedCalibrationDragLimit>;
}

type ActiveDrag = RotationDrag | TranslationDrag;

function vec3(vector: THREE.Vector3): StageVec3 {
  return [vector.x, vector.y, vector.z];
}

function eventRay(event: ThreeEvent<PointerEvent>): {
  readonly origin: StageVec3;
  readonly direction: StageVec3;
} {
  return {
    origin: vec3(event.ray.origin),
    direction: vec3(event.ray.direction),
  };
}

function localAxisQuaternion(axis: StageVec3): THREE.Quaternion {
  const direction = new THREE.Vector3(...axis).normalize();
  return new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    direction,
  );
}

function Handle({
  item,
  selected,
  hovered,
  disabled,
  onPointerOver,
  onPointerOut,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: {
  readonly item: JointHandle;
  readonly selected: boolean;
  readonly hovered: boolean;
  readonly disabled: boolean;
  readonly onPointerOver: (event: ThreeEvent<PointerEvent>) => void;
  readonly onPointerOut: (event: ThreeEvent<PointerEvent>) => void;
  readonly onPointerDown: (event: ThreeEvent<PointerEvent>) => void;
  readonly onPointerMove: (event: ThreeEvent<PointerEvent>) => void;
  readonly onPointerUp: (event: ThreeEvent<PointerEvent>) => void;
}) {
  const color = selected ? 0xffb020 : hovered ? 0x0a84ff : 0x78bdf2;
  return (
    <group position={item.world.pivot}>
      <mesh
        renderOrder={20}
        scale={selected ? 1.35 : hovered ? 1.16 : 1}
        onPointerOver={onPointerOver}
        onPointerOut={onPointerOut}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onLostPointerCapture={onPointerUp}
      >
        <sphereGeometry args={[0.028, 14, 10]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={disabled ? 0.35 : 0.82}
          depthTest={false}
        />
      </mesh>
      {selected && item.joint.type !== "prismatic" && (
        <mesh
          quaternion={localAxisQuaternion(item.world.axis)}
          renderOrder={19}
        >
          <torusGeometry args={[0.075, 0.007, 8, 48]} />
          <meshBasicMaterial
            color={0xffb020}
            transparent
            opacity={0.78}
            depthTest={false}
          />
        </mesh>
      )}
    </group>
  );
}

/** Declarative R3F joint handles shared by H2R and R2R calibration. */
export function ManipulatorLayer({
  interaction,
  onDraggingChange,
  onHoveredJointChange,
}: {
  readonly interaction: CalibrationInteractionModel;
  readonly onDraggingChange: (dragging: boolean) => void;
  readonly onHoveredJointChange: (name: string | null) => void;
}) {
  const root = useRef<THREE.Group | null>(null);
  const drag = useRef<ActiveDrag | null>(null);
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null);
  const joints = useMemo<readonly JointHandle[]>(
    () => interaction.jointLimits.flatMap((joint) => {
      const world = interaction.jointWorld[joint.name];
      return world?.pivot && world.axis ? [{ joint, world }] : [];
    }),
    [interaction.jointLimits, interaction.jointWorld],
  );

  useEffect(
    () => () => {
      const active = drag.current;
      if (active?.captureTarget.hasPointerCapture(active.pointerId)) {
        active.captureTarget.releasePointerCapture(active.pointerId);
      }
      drag.current = null;
      onHoveredJointChange(null);
      onDraggingChange(false);
    },
    [onDraggingChange, onHoveredJointChange],
  );

  function worldGeometry(item: JointHandle): {
    readonly pivot: StageVec3;
    readonly axis: StageVec3;
  } | null {
    const object = root.current;
    if (!object) return null;
    object.updateWorldMatrix(true, false);
    const pivot = new THREE.Vector3(...item.world.pivot).applyMatrix4(
      object.matrixWorld,
    );
    const axis = new THREE.Vector3(...item.world.axis)
      .transformDirection(object.matrixWorld)
      .normalize();
    return { pivot: vec3(pivot), axis: vec3(axis) };
  }

  function beginDrag(item: JointHandle, event: ThreeEvent<PointerEvent>): void {
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    if (event.button !== 0 || interaction.disabled) return;
    const geometry = worldGeometry(item);
    const captureTarget = event.target as unknown as R3fPointerCaptureTarget;
    if (!geometry) return;
    const ray = eventRay(event);
    const limit = resolvedCalibrationDragLimit(item.joint);
    const startValue = interaction.jointQ[item.joint.name] ?? 0;
    let next: ActiveDrag | null = null;
    if (item.joint.type === "prismatic") {
      const startParameter = jointAxisParameter(
        ray.origin,
        ray.direction,
        geometry.pivot,
        geometry.axis,
      );
      if (startParameter !== null) {
        next = {
          kind: "translation",
          pointerId: event.pointerId,
          captureTarget,
          joint: item.joint.name,
          pivot: geometry.pivot,
          axis: geometry.axis,
          startParameter,
          startValue,
          limit,
        };
      }
    } else {
      const point = jointPlanePoint(
        ray.origin,
        ray.direction,
        geometry.pivot,
        geometry.axis,
      );
      const startVector = point && jointDragVector(point, geometry.pivot);
      if (startVector) {
        next = {
          kind: "rotation",
          pointerId: event.pointerId,
          captureTarget,
          joint: item.joint.name,
          pivot: geometry.pivot,
          axis: geometry.axis,
          startVector,
          startValue,
          limit,
        };
      }
    }
    interaction.onSelectedJointChange(item.joint.name);
    if (!next) return;
    drag.current = next;
    captureTarget.setPointerCapture(event.pointerId);
    onDraggingChange(true);
  }

  function moveDrag(event: ThreeEvent<PointerEvent>): void {
    const active = drag.current;
    if (!active || active.pointerId !== event.pointerId) return;
    event.stopPropagation();
    const ray = eventRay(event);
    let value: number | null = null;
    if (active.kind === "translation") {
      const parameter = jointAxisParameter(
        ray.origin,
        ray.direction,
        active.pivot,
        active.axis,
      );
      if (parameter !== null) {
        value = calibrationDragValue(
          active.startValue,
          parameter - active.startParameter,
          active.limit,
        );
      }
    } else {
      const point = jointPlanePoint(
        ray.origin,
        ray.direction,
        active.pivot,
        active.axis,
      );
      const current = point && jointDragVector(point, active.pivot);
      if (current) {
        value = calibrationDragValue(
          active.startValue,
          signedJointDragAngle(active.startVector, current, active.axis),
          active.limit,
        );
      }
    }
    if (value !== null) interaction.onJointChange(active.joint, value);
  }

  function endDrag(event: ThreeEvent<PointerEvent>): void {
    const active = drag.current;
    if (!active || active.pointerId !== event.pointerId) return;
    event.stopPropagation();
    drag.current = null;
    if (active.captureTarget.hasPointerCapture(event.pointerId)) {
      active.captureTarget.releasePointerCapture(event.pointerId);
    }
    onDraggingChange(false);
    setHoveredJoint(null);
    onHoveredJointChange(null);
  }

  return (
    <group
      ref={root}
      name="calibration-manipulators"
      position-z={interaction.groundOffsetZ}
    >
      {joints.map((item) => (
        <Handle
          key={item.joint.name}
          item={item}
          selected={interaction.selectedJoint === item.joint.name}
          hovered={hoveredJoint === item.joint.name}
          disabled={Boolean(interaction.disabled)}
          onPointerOver={(event) => {
            event.stopPropagation();
            setHoveredJoint(item.joint.name);
            onHoveredJointChange(item.joint.name);
          }}
          onPointerOut={(event) => {
            event.stopPropagation();
            if (!drag.current) {
              setHoveredJoint(null);
              onHoveredJointChange(null);
            }
          }}
          onPointerDown={(event) => beginDrag(item, event)}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
        />
      ))}
    </group>
  );
}
