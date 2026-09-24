export type ViewId =
  | "motion"
  | "robot-assets"
  | "video-to-motion"
  | "h2r"
  | "r2r"
  | "batch"
  | "dataset-viz";

interface NavigationItem {
  id: ViewId;
  label: string;
  zhLabel: string;
  icon: string;
}

interface NavigationGroup {
  label: string;
  zhLabel: string;
  items: readonly NavigationItem[];
}

export const navigationGroups: readonly NavigationGroup[] = [
  {
    label: "Assets",
    zhLabel: "资源",
    items: [
      {
        id: "motion",
        label: "Motion",
        zhLabel: "动作",
        icon: "/icons/sidebar/motion.svg",
      },
      {
        id: "robot-assets",
        label: "Robot",
        zhLabel: "机器人",
        icon: "/icons/sidebar/robot.svg",
      },
    ],
  },
  {
    label: "Workflows",
    zhLabel: "工作流",
    items: [
      {
        id: "video-to-motion",
        label: "Video → Motion",
        zhLabel: "视频 → 动作",
        icon: "/icons/sidebar/video-to-motion.svg",
      },
      {
        id: "h2r",
        label: "Human → Robot",
        zhLabel: "人体 → 机器人",
        icon: "/icons/sidebar/h2r.svg",
      },
      {
        id: "r2r",
        label: "Robot → Robot",
        zhLabel: "机器人 → 机器人",
        icon: "/icons/sidebar/r2r.svg",
      },
      {
        id: "batch",
        label: "Batch",
        zhLabel: "批处理",
        icon: "/icons/sidebar/batch.svg",
      },
    ],
  },
  {
    label: "Analysis",
    zhLabel: "分析",
    items: [
      {
        id: "dataset-viz",
        label: "Data Analysis",
        zhLabel: "数据分析",
        icon: "/icons/sidebar/analysis.svg",
      },
    ],
  },
];
