# Robot tuning

**[中文](./robot-tuning_cn.md)**

Most robots work without manual tuning: drag a URDF and its meshes into the WebUI and joints are detected automatically. Use this page when a retarget needs adjusting.

## Where the files live

- Installed presets and Web uploads: `~/.config/hhtools/robots/<name>/robot.yaml`
- Field-by-field template: [`configs/robots/_template/robot.yaml`](../configs/robots/_template/robot.yaml)

**YAML edits apply on the next retarget,** with no Web restart. Restart `hhtools web` only after upgrading the Python package. Run `hhtools robot validate <name>` after editing. More background is in [framework.md](../framework.md).

Re-uploading a URDF regenerates `robot.yaml` from the URDF. Calibration files are kept, but a hand-edited `ik_map` or `weights` may be overwritten.

## `robot.yaml` fields

| Section | Purpose |
|---------|---------|
| `ik_map` | Canonical human joint → URDF link. On 3-DOF hips and shoulders, map to the **middle** link (usually `*_roll_link`). |
| `weights` | IK priorities: `t_weight` (position), `r_weight` (orientation). |
| [`smooth_joint_filter_masks`](#arms-stay-open-check-smooth_joint_filter_masks) | **High-impact IK regulariser.** Per-link midpoint pull, not the same as `weights`. |
| [`retarget.joint_scale_multipliers`](#adjust-proportions-joint_scale_multipliers) | Optional manual proportion overrides, without re-calibrating. |
| `retarget.feet_stabilizer`, `apply_feet_stabilizer` | Foot planting and body-ground clearance. Set `apply_feet_stabilizer: false` for rolls and flips. |
| `retarget.references.<format>` | Per motion-format overrides (for example a bundled `scaler_config`). |

## Arms stay open? Check `smooth_joint_filter_masks`

If arms stay in an A-pose while the mocap arms hang down, check this **before** tweaking `weights`.

These masks pair with the pipeline's default `smooth_joint_filter_weight: 5.5`. Per-link values in `[0, 1]` scale a *midpoint pull* on each joint. Scaffold defaults (`*_shoulder_roll_link: 1.0`) suit G1 and RP1-style gimbals, where roll is null-space. On uploaded URDFs whose **arm pose is driven mainly by shoulder roll**, `1.0` can lock the arms open and block tracking even when `weights` look correct.

**Lower roll to `0.1`–`0.3`** (or `0` for maximum arm freedom) if arms stay abducted while the yellow overlay hangs down. Keep pitch and yaw masks moderate for stability.

```yaml
smooth_joint_filter_masks:
  left_shoulder_pitch_link: 0.1
  left_shoulder_roll_link: 0.1   # not 1.0 when roll must move for arm tracking
  left_shoulder_yaw_link: 0.3
  right_shoulder_pitch_link: 0.1
  right_shoulder_roll_link: 0.1
  right_shoulder_yaw_link: 0.3
```

## Adjust proportions: `joint_scale_multipliers`

Per-canonical **absolute** scale overrides, in the same units as calibration `derived.scales`. For example, `left_shoulder: 0.5` narrows the upper body.

```yaml
retarget:
  joint_scale_multipliers:
    left_shoulder: 0.5
    right_shoulder: 0.5
    left_elbow: 1.0
    # … other ik_map keys; omit or leave at calibration values for no change
```

- Do **not** paste a calibration's `derived.scales` table here; it pollutes other human-reference formats.
- Values that match the current or any on-disk `retarget_calibration_*.yaml` scales, or leftover scaffold zero-pose defaults, are ignored.
- **Shoulders** affect lateral IK and shoulder roll only, not vertical height.
