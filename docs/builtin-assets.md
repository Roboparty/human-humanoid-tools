# Built-in release assets

This document is the human-readable view of
[`configs/builtin-assets.json`](../configs/builtin-assets.json). The JSON manifest is the
authoritative release allowlist; packaging must copy only declared paths, never walk an asset or
user-library directory recursively.

The motion baseline is commit `2c4de3980425ac10ff172a5724438fe1a8a076eb`, the repository state
used at the start of the 2026-09-08 refactor audit. It contains 31 discoverable motions. The
generated GVHMR result `assets/motions/mimic/GVHMR/hmr4d_results.pt` is explicitly rejected because
its motion loading is not reliable, leaving **30 bundled motions across 58 required files**.

## Human motions

| Profile | Dataset | Built-in entries |
| --- | --- | --- |
| `mimic` | AMASS | `B10_-_walk_turn_left_(45)_stageii` |
| `mimic` | GLB | `cranberry` |
| `mimic` | LAFAN | `dance1_subject2`; `fallAndGetUp1_subject1` |
| `mimic` | MOCAP | `D0429_侧空翻__01_Sk00_144`; `D0429_前空翻__01_Sk00_148`; `D0429_后空翻__01_Sk00_146`; `D0429_毽子后空翻__01_Sk00_149`; `D0429_踢月转体__01_Sk00_143` |
| `mimic` | Motion-X | `Aerial_Kick_Kungfu_wushu_3_clip2` |
| `mimic` | PHUMA | `kick` |
| `mimic` | SOMA | `big_light_one_hand_pick_up_front_low_R_005__A508` |
| `mimic` | Xsens_mocap | `stand`; `walk` |
| `intermimic` | OMOMO | `sub10_largebox_000`; `sub12_woodchair_000` |
| `intermimic` | OmniContact | `case1_carry_20260330000166_1_1775007076`; `case1_carry_20260330000166_1_1775009756` |
| `meshmimic` | holosoma | `parkour_1`; `parkour_2`; `parkour_3`; `parkour_4`; `parkour_5` |
| `meshmimic` | parc_ms | `BOXES_0_0_opt_dm_dm`; `BOXES_1_0_opt_dm_dm`; `beyond_reverse_vault_003_aug001_dm_aug8`; `beyond_speed_vault_005_aug007_flipped_dm_aug1`; `dec2024_teaser_0_0_opt_dm`; `dec2024_teaser_2000_0_opt_dm`; `dec2024_teaser_500_0_opt_dm` |

The manifest lists every required sidecar explicitly: OMOMO object meshes, OmniContact capture and
object files, holosoma heightfield sidecars and terrain meshes, parc_ms terrain meshes, and the
shared `holosoma/source.yaml`. Analysis caches under `.hhtools_analysis` are never release assets.

## Robots

| Preset | Display name | DoF | Pinned source |
| --- | --- | ---: | --- |
| `g1_29dof` | Unitree G1 | 29 | `unitreerobotics/unitree_ros@7d6075f7f585` |
| `roboto_origin` | ROBOTO_ORIGIN (RPO) | 23 | `Roboparty/rpo_description@37aac9ca665e` |
| `agibot_x2_ultra` | AgiBot X2 | 31 | `AgibotTech/agibot_x2_urdf@77f43eb0904d` |
| `asimov_1` | Asimov 1 | 23 | `menloresearch/asimov-1@b8420ffe9915` |
| `fourier_gr2` | Fourier GR-2 | 29 | `FFTAI/Wiki-GRx-Models@7d96c758f048` |
| `berkeley_humanoid_lite` | Berkeley Humanoid Lite | 22 | `HybridRobotics/Berkeley-Humanoid-Lite-Assets@fc90fedd008b` |

Each robot is installed by `scripts/install_builtin_robots.py` from a pinned upstream commit. A
release must include `SOURCE.json` plus only its `installed_files` records. Do not copy the entire
development Robot Library: it may contain user calibrations, replacement meshes, or custom robots.

## Verification

Validate the repository allowlist:

```bash
uv run python scripts/verify_builtin_assets.py
```

Also validate the currently installed six robot bundles and every payload hash:

```bash
uv run python scripts/verify_builtin_assets.py \
  --robot-root ~/.config/hhtools/robots
```

This allowlist controls selection, not legal clearance. Every release still needs a license review
for the selected motion data and pinned robot repositories.
