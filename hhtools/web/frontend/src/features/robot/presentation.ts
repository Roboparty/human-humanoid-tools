/** Shared catalog presentation for the library and workflow robot pickers. */

interface RobotIdentity {
  readonly name: string;
  readonly display_name: string;
}

const ROBOT_ORDER: Readonly<Record<string, number>> = {
  rp1: 0,
  g1_29dof: 1,
  roboto_origin: 2,
  agibot_x2_ultra: 3,
  asimov_1: 4,
  fourier_gr2: 5,
  berkeley_humanoid_lite: 6,
};

export const CURATED_LABELS: Readonly<Record<string, string>> = {
  rp1: "ROBOTO 01",
  g1_29dof: "Unitree G1",
  roboto_origin: "ROBOTO_ORIGIN (RPO)",
  agibot_x2_ultra: "AgiBot X2",
  asimov_1: "Asimov 1",
  fourier_gr2: "Fourier GR-2",
  berkeley_humanoid_lite: "Berkeley Humanoid Lite",
};

const ROBOT_ICONS: Readonly<Record<string, string>> = {
  rp1: "/robot-icons/rp1.png",
  g1_29dof: "/robot-icons/unitree-g1.webp",
  roboto_origin: "/robot-icons/roboto-origin.webp",
  agibot_x2_ultra: "/robot-icons/agibot-x2.webp",
  asimov_1: "/robot-icons/asimov-1.webp",
  fourier_gr2: "/robot-icons/fourier-gr2.webp",
  berkeley_humanoid_lite: "/robot-icons/berkeley-humanoid-lite.webp",
};

export const FALLBACK_ICON = "/hhtools-robot.svg";
export const CURATED_NAMES = Object.keys(CURATED_LABELS);

function mapValue<T>(map: Readonly<Record<string, T>>, key: string): T | undefined {
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined;
}

export function robotLabel(robot: RobotIdentity): string {
  return mapValue(CURATED_LABELS, robot.name) ?? (robot.display_name || robot.name);
}

export function sortRobots(left: RobotIdentity, right: RobotIdentity): number {
  const leftOrder = mapValue(ROBOT_ORDER, left.name);
  const rightOrder = mapValue(ROBOT_ORDER, right.name);
  if (leftOrder !== undefined || rightOrder !== undefined) {
    if (leftOrder === undefined) return 1;
    if (rightOrder === undefined) return -1;
    return leftOrder - rightOrder;
  }
  return robotLabel(left).localeCompare(robotLabel(right));
}

export function robotIcon(name: string): string {
  return mapValue(ROBOT_ICONS, name) ?? FALLBACK_ICON;
}
