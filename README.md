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

- **Fast retarget** — Web UI or **CLI** (`hhtools retarget` / `scripts/batch_*_retarget.py`); **Newton IK** + **MPC-SQP** interaction mesh.
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

### CLI (batch / no Web UI)

Entry point: `uv run hhtools` (same package as the Web UI). Use this for large datasets (thousands of clips) instead of dragging files into the browser. Calibrate once in the Web UI (or place `retarget_calibration_<ref>.yaml` next to the URDF) before batch retarget.

| Command | Purpose |
|---------|---------|
| `hhtools convert run` | BVH / GLB → unified NPZ |
| `hhtools import list` / `import run` | List adapters; import a dataset root → NPZ |
| `hhtools bodymodel check` / `setup` | SMPL-family weight paths / download hints |
| `hhtools robot list` / `info` / `schema` / `validate` / `scaffold` / `add` | Robot presets |
| `hhtools retarget run` | Newton IK → CSV (files or directory) |
| `hhtools retarget interaction-mesh run` | Interaction-mesh (terrain / objects) → CSV |
| `hhtools retarget interaction-mesh precompute-laplacian` | Precompute Laplacian targets (`.npz`) |
| `hhtools web` | HTML / three.js UI (default `127.0.0.1:8009`) |
| `hhtools ui` | Legacy Viser viewer |

**Convert & import**

```bash
uv run hhtools convert run assets/motions/mimic/LAFAN/dance1_subject2.bvh -o /tmp/npz --unit m
uv run hhtools convert run assets/motions/mimic/GLB/cranberry.glb -o /tmp/npz

uv run hhtools import list
uv run hhtools import run --dataset lafan \
  --root assets/motions/mimic/LAFAN -o /tmp/lafan_npz \
  --sequence dance1_subject2.bvh
uv run hhtools import run --dataset omomo \
  --root assets/motions/intermimic/OMOMO -o /tmp/omomo_npz \
  --sequence sub12_woodchair_000/sub12_woodchair_000.pkl
```

**Robots**

```bash
uv run hhtools robot list
uv run hhtools robot info unitree_g1__g1_29dof --no-mjcf
uv run hhtools robot schema unitree_g1__g1_29dof -o /tmp/g1_header.csv
uv run hhtools robot validate unitree_g1__g1_29dof
uv run hhtools robot scaffold unitree_g1          # skip existing yaml
# uv run hhtools robot add /path/to/urdf_or_dir  # ingest into configs/robots/
```

**Retarget (smoke with `--limit-frames`)**

```bash
# Newton IK (flat / AMASS-style NPZ)
uv run hhtools retarget run path/to/clip.npz \
  --robot unitree_g1__g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh (OMOMO / terrain clips)
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot unitree_g1__g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

**Large offline batches** (resumable, subprocess isolation; export matches Web CSV/sidecars, folders not zipped):

```bash
# mimic (flat mocap → Newton IK): amass | lafan | glb | …
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset amass \
  --in /path/to/AMASS --out /path/to/AMASS_rp1 \
  --skip-existing --limit 5

# intermimic (human–object): omomo
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omomo \
  --in /path/to/OMOMO --out /path/to/OMOMO_rp1 \
  --skip-existing

# meshmimic (terrain): parc_ms | holosoma
python scripts/batch_meshmimic_retarget.py \
  --robot rp1 --dataset parc_ms \
  --in /path/to/parc_ms --out /path/to/parc_ms_rp1 \
  --skip-existing --failure-log failures.jsonl

# robot→robot (input = already-exported source-robot trajectories)
python scripts/batch_r2r_retarget.py \
  --source-robot rp1 --target-robot unitree_g1__g1_29dof \
  --in /path/to/rp1_exports --out /path/to/g1_from_rp1 \
  --profile auto --skip-existing
```

Scene clips → `<out>/<clip>/<clip>.csv` + terrain/object sidecars (robot frame). Flat mimic → `<out>/…/<stem>.csv`. Interaction-mesh needs `mujoco` + `osqp`; Newton needs the NVIDIA `newton` package. R2R needs a saved `r2r_calibration_<source>.yaml` beside the target URDF (Web calibrate once, or `--calibration` / `--init-zero-calibration`).

### Tuning `robot.yaml`

Paths: bundled presets under `configs/robots/<name>/`; Web uploads under `~/.config/hhtools/robots/<name>/`. **Yaml edits apply on the next retarget** (no Web restart). Restart `hhtools web` only after upgrading the Python package.

| Section | Purpose |
|---------|---------|
| `ik_map` | Canonical human joint → URDF link. On 3-DOF hips/shoulders, map to the **middle** link (usually `*_roll_link`). |
| `weights` | IK priorities: `t_weight` (position), `r_weight` (orientation). |
| **`smooth_joint_filter_masks`** | **High-impact IK regulariser** (pairs with default `smooth_joint_filter_weight: 5.5` in the pipeline). Per-link values in `[0, 1]` scale a *midpoint pull* on each joint — **not** the same as `weights`. Scaffold defaults (`*_shoulder_roll_link: 1.0`) suit G1/RP1-style gimbals where roll is null-space; on uploaded URDFs whose **arm pose is driven mainly by shoulder roll**, **`1.0` can lock the arms open** and block tracking even when `weights` look correct. **Lower roll to `0.1`–`0.3`** (or `0` for max arm freedom) if retarget arms stay abducted while the yellow overlay hangs down; keep pitch/yaw masks moderate for stability. |
| `retarget.joint_scale_multipliers` | Optional. Per-canonical **absolute** scale overrides (same units as calibration `derived.scales`) for **manual** proportion tweaks without re-calibrating. Example: `left_shoulder: 0.5` narrows the upper body. Do **not** paste a calibration's `derived.scales` table here (it pollutes other human-reference formats). Values that match the **current or any** on-disk `retarget_calibration_*.yaml` scales (or leftover scaffold zero-pose defaults) are ignored. **Shoulders** affect lateral IK + shoulder roll only (not vertical height). |
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
