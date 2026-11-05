# 机器人调参

**[English](robot-tuning.md)**

多数机器人无需手动调参：在 WebUI 中拖入 URDF 和 mesh，即可自动识别关节。重映射效果需要调整时，再参考本页。

## 文件位置

- 已安装的预设与 Web 上传的机器人：`~/.config/hhtools/robots/<名称>/robot.yaml`
- 逐字段说明的模板：[`configs/robots/_template/robot.yaml`](../configs/robots/_template/robot.yaml)

**修改 yaml 后，下次重映射即生效，** 无需重启 Web；仅在升级 Python 包后需要重启 `hhtools web`。修改后可运行 `hhtools robot validate <名称>` 检查。原理见 [framework.md](../framework.md)。

重新上传 URDF 会按 URDF 重新生成 `robot.yaml`。标定文件会保留，但手动修改过的 `ik_map` 或 `weights` 可能被覆盖。

## `robot.yaml` 字段

| 字段 | 作用 |
|------|------|
| `ik_map` | 标准人体关节 → URDF link。三自由度的髋和肩应映射到**中间**的 link（多为 `*_roll_link`）。 |
| `weights` | IK 权重：`t_weight` 为位置，`r_weight` 为朝向。 |
| [`smooth_joint_filter_masks`](#手臂一直张开先检查-smooth_joint_filter_masks) | **对姿态影响很大的 IK 正则项。** 按 link 把关节拉向中点，与 `weights` 不同。 |
| [`retarget.joint_scale_multipliers`](#调整体型joint_scale_multipliers) | 可选，手动覆盖体型比例，无需重新标定。 |
| `retarget.feet_stabilizer`、`apply_feet_stabilizer` | 脚底贴地与身体离地高度。翻滚、空翻类动作可设 `apply_feet_stabilizer: false`。 |
| `retarget.references.<格式>` | 按动作格式覆盖（例如 bundled `scaler_config`）。 |

## 手臂一直张开？先检查 `smooth_joint_filter_masks`

如果动捕中手臂下垂，机器人手臂却一直呈 A 字张开，请**先查此项**，不要只改 `weights`。

这些系数与 pipeline 默认的 `smooth_joint_filter_weight: 5.5` 配合使用。每个 link 的值在 `[0, 1]` 之间，用来把关节往限位**中点**拉。脚手架默认值（如 `*_shoulder_roll_link: 1.0`）适合 G1、RP1 这类 roll 主要处于零空间的万向节；但对于**手臂主要靠 shoulder roll 才能垂下**的上传 URDF，`1.0` 会把手臂锁在张开位，即使 `weights` 已调高也无法跟踪。

黄色骨架已下垂、机器人仍张臂时，**把 roll 降到 `0.1`–`0.3`**（需要最大自由度时可设为 `0`）；pitch 和 yaw 保持中等值以防抖动。

```yaml
smooth_joint_filter_masks:
  left_shoulder_pitch_link: 0.1
  left_shoulder_roll_link: 0.1   # 需要 roll 参与摆臂时不要用 1.0
  left_shoulder_yaw_link: 0.3
  right_shoulder_pitch_link: 0.1
  right_shoulder_roll_link: 0.1
  right_shoulder_yaw_link: 0.3
```

## 调整体型：`joint_scale_multipliers`

各标准关节的**绝对**缩放覆盖值，单位与标定文件中的 `derived.scales` 相同。例如 `left_shoulder: 0.5` 会收窄上半身。

```yaml
retarget:
  joint_scale_multipliers:
    left_shoulder: 0.5
    right_shoulder: 0.5
    left_elbow: 1.0
    # … 其余 ik_map 关节；与标定一致的可以省略
```

- **不要**把某份标定的 `derived.scales` 整表贴进来，否则会影响其他人体参考格式。
- 与当前或任意 `retarget_calibration_*.yaml` 的 scales 相同的值（或仍是脚手架零位默认值）会被视为未修改并忽略。
- **肩**只影响横向 IK 与 shoulder roll，不改变竖直身高。
