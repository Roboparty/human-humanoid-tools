# human-humanoid-tools (hhtools)

**Retarget parkour, dance, and interaction clips onto any humanoid in ~30 seconds**

**[Project page](https://roboparty.github.io/human-humanoid-tools/)** · **[中文说明](README_cn.md)**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Roboparty%2Fhuman--humanoid--tools-blue)](https://github.com/Roboparty/human-humanoid-tools)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue)](https://roboparty.github.io/human-humanoid-tools/)

| | |
| :---: | :---: |
| ![](assets/readme/demo-01.gif) | ![](assets/readme/demo-02.gif) |
| ![](assets/readme/demo-03.gif) | ![](assets/readme/demo-04.gif) |

---

We welcome suggestions and ideas — please open an issue or discussion anytime. New feature requests will be considered once the core functionality is stable.

---

## Highlights

- **Fast retarget** — drag a human clip, pick a robot, export CSV/ZIP; **Newton IK** + **MPC-SQP** interaction mesh.
- **Human formats** — BVH / GLB / SMPL family; adapters for AMASS, GVHMR, LAFAN, OMOMO, PHUMA, intermimic, meshmimic, …
- **Any URDF** — upload any robot in the Web UI: drag in the URDF, drag in meshes; auto-detected, no manual tuning.
- **Robot→robot (R2R)** — retarget existing robot CSV/PKL exports onto a new URDF.
- **Dataset analysis** — scan, tag, embed, cluster, and subset human or robot motion libraries in the Web UI.

**Requirements:** Linux, Python 3.12+. Preview on CPU; retarget needs **NVIDIA GPU (CUDA 12)**.

---

## Quick start

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv sync --extra all
uv run hhtools web
```

Open `http://127.0.0.1:8009`.

| Panel | Flow |
|-------|------|
| **Motion → Robot** | Load clip → select robot → calibrate (once) → retarget → download CSV/ZIP |
| **Robot → Robot** | Source robot + trajectory → target URDF → calibrate → retarget / batch ZIP |
| **Dataset analysis** | Drop a folder → analyze → explore tags & scatter → export subset |

Robot tuning: edit [`configs/robots/unitree_g1/`](configs/robots/unitree_g1/) or uploaded `~/.config/hhtools/robots/<name>/robot.yaml`; run `hhtools robot validate <name>`. Details in [framework.md](framework.md).

### Tuning `robot.yaml`

Paths: bundled presets under `configs/robots/<name>/`; Web uploads under `~/.config/hhtools/robots/<name>/`. **Yaml edits apply on the next retarget** (no Web restart). Restart `hhtools web` only after upgrading the Python package.

| Section | Purpose |
|---------|---------|
| `ik_map` | Canonical human joint → URDF link. On 3-DOF hips/shoulders, map to the **middle** link (usually `*_roll_link`). |
| `weights` | IK priorities: `t_weight` (position), `r_weight` (orientation). |
| **`smooth_joint_filter_masks`** | **High-impact IK regulariser** (pairs with default `smooth_joint_filter_weight: 5.5` in the pipeline). Per-link values in `[0, 1]` scale a *midpoint pull* on each joint — **not** the same as `weights`. Scaffold defaults (`*_shoulder_roll_link: 1.0`) suit G1/RP1-style gimbals where roll is null-space; on uploaded URDFs whose **arm pose is driven mainly by shoulder roll**, **`1.0` can lock the arms open** and block tracking even when `weights` look correct. **Lower roll to `0.1`–`0.3`** (or `0` for max arm freedom) if retarget arms stay abducted while the yellow overlay hangs down; keep pitch/yaw masks moderate for stability. |
| `retarget.joint_scale_multipliers` | Per-canonical **absolute** scale (same units as calibration `derived.scales`). Written after calibration; edit individual keys to fine-tune body proportions without re-calibrating. Example: `left_shoulder: 0.5` narrows the upper body. **Shoulders** affect lateral IK + shoulder roll only (not vertical height). Values equal to calibration are ignored. |
| `retarget.feet_stabilizer`, `apply_feet_stabilizer` | Foot planting and body-ground clearance; set `apply_feet_stabilizer: false` for rolls / flips. |
| `retarget.references.<format>` | Per motion-format overrides (e.g. bundled `scaler_config`). |

```yaml
retarget:
  joint_scale_multipliers:
    left_shoulder: 0.5
    right_shoulder: 0.5
    left_elbow: 1.0
    # … other ik_map keys; omit or leave at calibration values for no change
```

**`smooth_joint_filter_masks` example** — if arms stay in an A-pose while mocap arms hang down, check this *before* only tweaking `weights`:

```yaml
smooth_joint_filter_masks:
  left_shoulder_pitch_link: 0.1
  left_shoulder_roll_link: 0.1   # not 1.0 when roll must move for arm tracking
  left_shoulder_yaw_link: 0.3
  right_shoulder_pitch_link: 0.1
  right_shoulder_roll_link: 0.1
  right_shoulder_yaw_link: 0.3
```

Template and field notes: [`configs/robots/_template/robot.yaml`](configs/robots/_template/robot.yaml). Re-uploading a URDF regenerates `robot.yaml` from the URDF (calibration files are kept; hand-edited `ik_map` / weights may be overwritten).

---

## Demo clips (`assets/motions`)

Demo paths only — download full datasets from upstream. Adapters provided; **no dataset redistribution**.

| Mode | Dataset | Paper | Download |
|------|---------|-------|----------|
| mimic | AMASS | [arXiv](https://arxiv.org/abs/1904.03278) | [site](https://amass.is.tue.mpg.de/) |
| mimic | GVHMR | [arXiv](https://arxiv.org/abs/2409.06662) | [GitHub](https://github.com/zju3dv/GVHMR) |
| mimic | LAFAN1 | [arXiv](https://arxiv.org/abs/2102.04942) | [GitHub](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) |
| mimic | Motion-X | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf) | [GitHub](https://github.com/IDEA-Research/Motion-X) |
| mimic | PHUMA | [arXiv](https://arxiv.org/abs/2510.26236) | [GitHub](https://github.com/DAVIAN-Robotics/PHUMA) |
| mimic | SOMA | [arXiv](https://arxiv.org/abs/2603.16858) | [Hugging Face](https://huggingface.co/datasets/bones-studio/seed) |
| intermimic | OMOMO | [arXiv](https://arxiv.org/abs/2309.16237) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_omomo) |
| meshmimic | holosoma | [arXiv](https://arxiv.org/abs/2509.26633) | [GitHub](https://github.com/amazon-far/holosoma) |
| meshmimic | PARC MS | [arXiv](https://arxiv.org/abs/2505.04002) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_parc_ms) |

---

## Citation

If you use **human-humanoid-tools** in research or products, please cite the repository:

```bibtex
@software{human_humanoid_tools2026,
  title        = {human-humanoid-tools (hhtools): humanoid motion retargeting and dataset analysis},
  author       = {jaggerShen and hhtools contributors},
  year         = {2026},
  url          = {https://github.com/Roboparty/human-humanoid-tools},
  license      = {Apache-2.0}
}
```

**Links:** [GitHub repository](https://github.com/Roboparty/human-humanoid-tools) · [Issues](https://github.com/Roboparty/human-humanoid-tools/issues) · [LICENSE](LICENSE)

When publishing results built on bundled adapters, also cite the **upstream datasets and solvers** listed above and in [NOTICE](NOTICE) (e.g. SOMA-Retargeter, holosoma).

---

## License & assets

- **Code:** [Apache-2.0](LICENSE) · third-party: [NOTICE](NOTICE)
- **SMPL / SMPL-H / SMPL-X weights:** not included; register at MPI and place under `configs/body_models/` — see [configs/body_models/README.md](configs/body_models/README.md)
- **More docs:** [framework.md](framework.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
