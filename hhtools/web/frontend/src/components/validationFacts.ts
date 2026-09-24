import {
  isNearCalibrationLimit,
  resolveCalibrationJointLimits,
  type CalibrationJointLimit,
} from "./calibrationEditorState.ts";
import type { ValidationItem } from "./ValidationSummary";
import { referenceTargetLink } from "../stage/referenceSkeleton.ts";
import type { StageMotionPayload, StageRobotPayload } from "../stage/types.ts";

type LocaleText = (english: string, chinese: string) => string;

const englishText: LocaleText = (english) => english;

function finitePositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

export function motionValidationFacts(
  motion: StageMotionPayload | null,
  text: LocaleText = englishText,
): readonly ValidationItem[] {
  if (!motion) return [];
  const frames = motion.positions.length || motion.playback_frames || 0;
  const fps = finitePositive(motion.framerate)
    ? motion.framerate
    : finitePositive(motion.sample_rate)
      ? motion.sample_rate
      : null;
  const joints = motion.parent_indices.length;
  const frameJoints = motion.positions[0]?.length ?? 0;
  const sceneParts = [
    motion.terrain ? text("terrain", "地形") : "",
    motion.objects?.length
      ? text(
          `${motion.objects.length} interaction object${motion.objects.length === 1 ? "" : "s"}`,
          `${motion.objects.length} 个交互物体`,
        )
      : "",
  ].filter(Boolean);
  return [
    {
      tone: frames > 0 ? "ok" : "error",
      label:
        frames > 0
          ? text(
              `Playable trajectory: ${frames} frames`,
              `可播放轨迹：${frames} 帧`,
            )
          : text("No playable frames", "没有可播放帧"),
    },
    {
      tone: fps ? "ok" : "warning",
      label: fps
        ? text(
            `Timeline: ${fps.toFixed(1)} FPS`,
            `时间轴：${fps.toFixed(1)} FPS`,
          )
        : text("Frame rate was not detected", "未检测到帧率"),
    },
    {
      tone: joints > 0 && frameJoints >= joints ? "ok" : "error",
      label:
        joints > 0 && frameJoints >= joints
          ? text(`Skeleton: ${joints} joints`, `骨架：${joints} 个关节`)
          : text(
              "Skeleton hierarchy and positions do not match",
              "骨架层级与关节位置不匹配",
            ),
    },
    {
      tone: motion.body_mesh?.available ? "ok" : "neutral",
      label: motion.body_mesh?.available
        ? text("Body surface is available", "身体表面可用")
        : `${text("Body uses the compact fallback", "身体使用精简回退模型")}${
            motion.body_mesh?.reason ? `: ${motion.body_mesh.reason}` : ""
          }`,
    },
    {
      tone: sceneParts.length ? "ok" : "neutral",
      label: sceneParts.length
        ? text(
            `Scene: ${sceneParts.join(" + ")}`,
            `场景：${sceneParts.join(" + ")}`,
          )
        : text("No interaction scene", "没有交互场景"),
    },
  ];
}

export function robotValidationFacts(
  robot: StageRobotPayload | null,
  text: LocaleText = englishText,
): readonly ValidationItem[] {
  if (!robot) return [];
  const mappings = Object.entries(robot.ik_map ?? {});
  const links = new Set(robot.links);
  const unresolved = mappings.flatMap(([, target]) => {
    const link = referenceTargetLink(target);
    return link && !links.has(link) ? [link] : [];
  });
  const dof = robot.num_dof ?? robot.actuated_joints?.length ?? 0;
  return [
    {
      tone: dof > 0 ? "ok" : "error",
      label: text(`${dof} controllable DoF`, `${dof} 个可控自由度`),
    },
    {
      tone: mappings.length ? "ok" : "warning",
      label: text(
        `Semantic map: ${mappings.length}/17 slots`,
        `语义映射：${mappings.length}/17 个位置`,
      ),
    },
    {
      tone: unresolved.length ? "error" : "ok",
      label: unresolved.length
        ? text(
            `${unresolved.length} mapped links are unresolved`,
            `${unresolved.length} 个映射连杆无法解析`,
          )
        : text("All mapped links resolve", "所有映射连杆均可解析"),
    },
    {
      tone: robot.glb_base64 ? "ok" : "warning",
      label: robot.glb_base64
        ? text(
            `Renderable model: ${robot.links.length} links`,
            `可渲染模型：${robot.links.length} 个连杆`,
          )
        : text(
            "Robot mesh is unavailable; link fallback will be shown",
            "机器人网格不可用，将显示连杆回退模型",
          ),
    },
  ];
}

export function calibrationValidationFacts(
  robot: StageRobotPayload | null,
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>>,
  text: LocaleText = englishText,
): readonly ValidationItem[] {
  if (!robot) return [];
  const base = robotValidationFacts(robot, text).slice(1, 3);
  const resolved = resolveCalibrationJointLimits(limits, values);
  const changed = resolved.filter((limit) => Math.abs(values[limit.name] ?? 0) > 1e-4);
  const near = resolved.filter((limit) =>
    isNearCalibrationLimit(values[limit.name] ?? 0, limit),
  );
  return [
    ...base,
    {
      tone: changed.length ? "ok" : "neutral",
      label: text(
        `${changed.length}/${resolved.length} joints differ from URDF zero`,
        `${changed.length}/${resolved.length} 个关节偏离 URDF 零位`,
      ),
    },
    {
      tone: near.length ? "warning" : "ok",
      label: near.length
        ? text(
            `${near.length} joints are near their limits`,
            `${near.length} 个关节接近限位`,
          )
        : text("No joints are near their limits", "没有关节接近限位"),
    },
  ];
}
