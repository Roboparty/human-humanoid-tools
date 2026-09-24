"""Render deterministic Robot Library thumbnails from installed URDF presets.

The renderer intentionally uses a small software projection instead of an
OpenGL window, so maintainers can reproduce the checked-in WebP files in a
headless build environment. See ``robot-icons/ATTRIBUTION.md`` for upstream
model revisions and the license that applies to each rendered thumbnail.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from yourdfpy import URDF


@dataclass(frozen=True)
class RobotIconRecipe:
    urdf: str
    accent: str


RECIPES: dict[str, RobotIconRecipe] = {
    "unitree-g1": RobotIconRecipe("g1_29dof/g1_29dof.urdf", "#1677ff"),
    "roboto-origin": RobotIconRecipe("roboto_origin/rpo.urdf", "#8b5cf6"),
    "agibot-x2": RobotIconRecipe("agibot_x2_ultra/X2-Ultra.urdf", "#0ea5a4"),
    "asimov-1": RobotIconRecipe("asimov_1/asimov_1.urdf", "#e07a20"),
    "fourier-gr2": RobotIconRecipe("fourier_gr2/gr2v3_8_7.urdf", "#e34b4b"),
    "berkeley-humanoid-lite": RobotIconRecipe(
        "berkeley_humanoid_lite/berkeley_humanoid_lite.urdf",
        "#2e9d62",
    ),
}


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def render_icon(urdf_path: Path, accent_hex: str, output_path: Path) -> None:
    """Render one zero-pose URDF as a 128 px, transparent, lossless WebP."""

    robot = URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)
    mesh = robot.scene.to_geometry()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # Add a bounded sample of triangle centres so broad, low-poly surfaces do
    # not become sparse after projection. The sampling is deterministic.
    step = max(1, len(faces) // 220_000)
    centres = vertices[faces[::step]].mean(axis=1)
    points = np.concatenate((vertices, centres), axis=0)

    centre = (points.min(axis=0) + points.max(axis=0)) * 0.5
    # HHTools robot presets use Z-up coordinates. A fixed elevated three-quarter
    # view preserves the humanoid silhouette while showing limb depth.
    view_from = np.array([2.3, -3.0, 0.65], dtype=np.float64)
    view_from /= np.linalg.norm(view_from)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    screen_right = np.cross(world_up, view_from)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(view_from, screen_right)
    screen_up /= np.linalg.norm(screen_up)

    relative = points - centre
    projected_x = relative @ screen_right
    projected_y = relative @ screen_up
    depth = relative @ view_from

    width = float(projected_x.max() - projected_x.min())
    height = float(projected_y.max() - projected_y.min())
    scale = 102.0 / max(width, height)
    pixel_x = np.rint(64.0 + projected_x * scale).astype(np.int32)
    pixel_y = np.rint(64.0 - projected_y * scale).astype(np.int32)
    inside = (pixel_x >= 0) & (pixel_x < 128) & (pixel_y >= 0) & (pixel_y < 128)
    pixel_x, pixel_y, depth = pixel_x[inside], pixel_y[inside], depth[inside]

    # Keep the camera-nearest point in each output pixel.
    linear = pixel_y * 128 + pixel_x
    order = np.lexsort((depth, linear))
    sorted_linear = linear[order]
    is_last = np.r_[sorted_linear[1:] != sorted_linear[:-1], True]
    chosen = order[is_last]

    depth_low, depth_high = np.percentile(depth[chosen], [2.0, 98.0])
    depth_span = max(float(depth_high - depth_low), 1e-9)
    shade = np.clip((depth[chosen] - depth_low) / depth_span, 0.0, 1.0)

    accent = np.array(_hex_rgb(accent_hex), dtype=np.float64)
    dark = np.array([38.0, 43.0, 52.0], dtype=np.float64)
    light = np.array([232.0, 238.0, 244.0], dtype=np.float64)
    colours = dark[None, :] * (1.0 - shade[:, None]) + light[None, :] * shade[:, None]
    colours = colours * 0.82 + accent[None, :] * 0.18

    pixels = np.zeros((128, 128, 4), dtype=np.uint8)
    pixels[pixel_y[chosen], pixel_x[chosen], :3] = np.rint(colours).astype(np.uint8)
    pixels[pixel_y[chosen], pixel_x[chosen], 3] = 255
    detail_layer = Image.fromarray(pixels)

    # Close sub-pixel gaps while retaining the model's projected silhouette.
    alpha = detail_layer.getchannel("A").filter(ImageFilter.MaxFilter(3))
    silhouette = Image.new("RGBA", (128, 128), (48, 54, 64, 255))
    silhouette.putalpha(alpha)
    robot_layer = Image.alpha_composite(silhouette, detail_layer)

    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    accent_rgb = _hex_rgb(accent_hex)
    draw.rounded_rectangle(
        (7, 7, 120, 120),
        radius=26,
        fill=(*accent_rgb, 24),
        outline=(*accent_rgb, 64),
        width=2,
    )
    canvas.alpha_composite(robot_layer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "WEBP", lossless=True, method=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-root",
        type=Path,
        help="Robot Library root; defaults to the current HHTools user library",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "hhtools"
        / "web"
        / "frontend"
        / "public"
        / "robot-icons",
    )
    args = parser.parse_args()

    if args.robot_root is None:
        from hhtools.utils.paths import user_robot_dir

        args.robot_root = user_robot_dir()

    for name, recipe in RECIPES.items():
        urdf_path = args.robot_root / recipe.urdf
        if not urdf_path.is_file():
            raise FileNotFoundError(f"missing curated robot URDF: {urdf_path}")
        output_path = args.output_dir / f"{name}.webp"
        render_icon(urdf_path, recipe.accent, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
