export type StageVec3 = readonly [number, number, number];
export type StageQuaternion = readonly [number, number, number, number];
export type StageMatrix4 = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export type R2rStageLayerId =
  | "r2r-source-robot"
  | "r2r-source-skeleton"
  | "r2r-source-scene"
  | "r2r-target-robot"
  | "r2r-target-skeleton"
  | "r2r-target-scene";

export type StageLayerId =
  | "skeleton"
  | "body"
  | "objects"
  | "scaled-skeleton"
  | "scaled-scene"
  | "robot"
  | R2rStageLayerId;

export type R2rLayerAvailability = Readonly<Record<R2rStageLayerId, boolean>>;

export interface StageTerrainPayload {
  readonly vertices: readonly StageVec3[];
  readonly faces: readonly (readonly [number, number, number])[];
}

export interface StageObjectPayload {
  readonly name?: string;
  readonly color?: readonly [number, number, number] | null;
  readonly extents: StageVec3;
  readonly opacity?: number | null;
  readonly positions: readonly StageVec3[];
  readonly quaternions: readonly StageQuaternion[];
  readonly has_mesh?: boolean;
  readonly source_index?: number;
  readonly scale?: number;
  readonly mesh_file?: string;
}

export interface StageBodyMeshPayload {
  readonly available: boolean;
  readonly type?: string;
  readonly vertices_gz_b64?: string;
  readonly num_verts?: number;
  readonly num_frames?: number;
  readonly triangles?: readonly (readonly [number, number, number])[];
  readonly reason?: string;
}

/** Data-only motion shape consumed by R3F layer components. */
export interface StageMotionPayload {
  readonly name?: string;
  readonly token?: string;
  readonly positions: readonly (readonly StageVec3[])[];
  readonly parent_indices: readonly number[];
  readonly exclude_joint_indices?: readonly number[];
  readonly frame_indices?: readonly number[];
  readonly playback_duration?: number;
  readonly duration?: number;
  readonly framerate?: number;
  readonly sample_rate?: number;
  readonly playback_frames?: number;
  readonly num_frames_total?: number;
  readonly source_format?: string;
  readonly up_axis?: string;
  readonly bone_names?: readonly string[];
  readonly canonical_names?: readonly string[];
  readonly quaternions?: readonly (readonly StageQuaternion[])[];
  readonly color?: number;
  readonly objects?: readonly StageObjectPayload[];
  readonly terrain?: StageTerrainPayload | null;
  readonly has_terrain?: boolean;
  readonly body_mesh?: StageBodyMeshPayload;
  readonly dataset?: string;
  readonly suggested_reference?: string;
  readonly suggested_backend?: string;
  readonly meta?: Readonly<Record<string, unknown>>;
  readonly library_entry?: {
    readonly source_path: string;
  };
  readonly object_mesh_source?: {
    readonly kind: "motion" | "r2r" | "dataset";
    readonly token: string;
  };
}

/** Zero-pose robot data returned by `/api/robot/select`. */
export interface StageRobotPayload {
  readonly name: string;
  readonly display_name: string;
  readonly base_link?: string;
  readonly links: readonly string[];
  readonly actuated_joints?: readonly string[];
  readonly num_dof?: number;
  readonly ik_map?: Readonly<Record<string, unknown>>;
  readonly ik_prewarmed?: boolean;
  readonly link_transforms_zero: Readonly<Record<string, StageMatrix4>>;
  readonly mesh_to_link?: Readonly<Record<string, string>>;
  readonly glb_base64?: string | null;
  readonly ground_offset_z?: number;
}

export interface StageRobotFrame {
  readonly root?: readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  readonly mesh_z_lift?: number;
  readonly links: Readonly<Record<string, StageMatrix4>>;
}

/** Per-frame FK payload returned by H2R and R2R jobs. */
export interface StageRobotTrajectoryPayload {
  readonly frames: readonly StageRobotFrame[];
  readonly frame_indices?: readonly number[];
  readonly duration?: number;
  readonly playback_duration?: number;
  readonly playback_frames?: number;
  readonly num_frames_total?: number;
  readonly framerate?: number;
  readonly sample_rate?: number;
}

export type StageTimelinePayload =
  | StageMotionPayload
  | StageRobotTrajectoryPayload;

/** One side of the R2R comparison surface. */
export interface StageR2rActorPayload {
  readonly robot: StageRobotPayload | null;
  readonly trajectory: StageRobotTrajectoryPayload | null;
  readonly skeleton: StageMotionPayload | null;
  /** Scene-only motion payload carrying terrain, objects, and their mesh token. */
  readonly environment: StageMotionPayload | null;
}

/** Complete data-only description of the two-sided R2R Stage. */
export interface StageR2rPresentationPayload {
  readonly phase: "source" | "calibration" | "result";
  readonly source: StageR2rActorPayload;
  readonly target: StageR2rActorPayload;
  readonly calibrationReference: StageMotionPayload | null;
  /** Stable identities let Stage reset defaults only for genuinely new results. */
  readonly sourceToken: string | null;
  readonly resultToken: string | null;
}
