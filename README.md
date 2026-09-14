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
- **Human formats** — BVH / GLB / SMPL family; adapters for AMASS, GVHMR, LAFAN, 100STYLE, OMOMO, OmniContact, PHUMA, intermimic, meshmimic, …
- **Any URDF** — upload any robot in the Web UI: drag in the URDF, drag in meshes; auto-detected, no manual tuning.
- **Robot→robot (R2R)** — retarget existing robot CSV/PKL exports onto a new URDF, including [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) G1 CSVs.
- **Dataset analysis** — scan, tag, embed, cluster, and subset human or robot motion libraries in the Web UI.

**Requirements:** Python 3.12+. Linux/Windows support **NVIDIA GPU (CUDA 12)** acceleration;
the Apple Silicon macOS desktop package uses CPU retargeting. See the desktop section for packaging requirements.
Video-to-motion additionally needs a separate CUDA-capable GVHMR installation.

---

## Install and run

hhtools has three interactive modes plus an Agent automation interface. They share the same motion,
robot, and retargeting core, but their installation and launch paths are intentionally separate:

| Mode | Best for | Launch |
|------|----------|--------|
| **Terminal (CLI/TUI workflow)** | Batch jobs, servers, SSH, and automation | `uv run hhtools ...` |
| **WebUI** | Browser-based visualization and interactive workflows | `uv run hhtools web` |
| **Desktop GUI** | Windows standalone app or Linux / macOS first-run setup | Application menu, Mac Applications, or `hhtools-desktop` |
| **Agent (JSON CLI / MCP)** | Versioned H2R, scene-free R2R, and scalable Batch automation | [`hhtools agent` / `hhtools-mcp`](docs/agent.md) |

### Recommended: source checkout with uv

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository.
HHTools supports Python 3.12 or newer; uv selects a compatible interpreter and can install one when
needed:

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
# Only needed when no compatible Python is installed:
uv python install 3.12
```

For the complete HHTools environment—formats, robots, retargeting, WebUI, and Agent/MCP—use the
project-maintained `all` extra:

```bash
uv sync --locked --extra all
uv run hhtools --help
uv run hhtools web
```

Open `http://127.0.0.1:8009`. GVHMR and separately licensed SMPL-family model weights remain
external; the `all` extra installs their Python integration libraries, not those model files.

For a smaller environment, install only the workflow you need. The core terminal commands require
no extra:

```bash
uv sync --locked
uv run hhtools --help
```

For WebUI preview and Newton retargeting:

```bash
uv sync --locked --extra web --extra retarget
uv run hhtools web
```

For a preview-only WebUI, omit `--extra retarget`. If a required package is absent, startup exits
with the missing package names and the exact recovery command instead of an import traceback.

### Agent and MCP

HHTools provides a strict JSON CLI for scripts and a local stdio MCP server for compatible agents.
The current Agent interface covers safe, preflighted H2R, scene-free R2R, scalable H2R/R2R
Batch jobs, and content-bound calibration assistance with GPT-visible front/side previews and
validated silent save. It does not yet expose the full WebUI feature set. See
[Agent interfaces](docs/agent.md) for installation, scope, the smoke-first workflow, runtime
ownership, and the included Codex project configuration.

### Desktop GUI

Download the **0.1.0 preview** desktop package for your platform:

| Platform | Download | Architecture |
|----------|----------|--------------|
| Windows | [EXE installer](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-x64-setup.exe) | x64 |
| Linux | [Debian package](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-amd64.deb) | amd64 |
| macOS 12+ | [DMG installer](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-mac-arm64.dmg) | Apple Silicon (arm64) |

The Debian package contains Electron plus the curated built-in motions and robots. On first launch,
its setup page uses the HHTools wheel, locked dependency list, uv configuration, and uv executable
carried inside that same package. The recommended per-user option needs no administrator password,
while the all-users option opens the operating system authentication dialog:

```bash
sudo apt install ./hhtools-0.1.0-amd64.deb
hhtools-desktop
```

The Windows installer instead bundles its Python runtime and application source, so it starts
without a checkout or system Python.

**macOS installation:**

1. Download and open the DMG above, then drag **Human-Humanoid Tools** into **Applications**.
2. Launch the app from Applications. This preview is ad-hoc signed and has not been notarized by
   Apple; if macOS blocks it, approve this app under **System Settings → Privacy & Security → Open Anyway**.
3. Follow the first-run setup and keep an internet connection while it downloads an isolated
   Python 3.12 environment. No existing Python, source checkout, or Homebrew installation is needed;
   the runtime is installed for the current user without an administrator password.

The macOS package includes the same 30 built-in motions and six robots as the other desktop packages.
Warp retargeting runs on CPU; Intel Macs and the local NVIDIA GVHMR pipeline are unsupported.

GVHMR and separately licensed SMPL-family weights remain optional on all platforms. Build details are in
[`desktop/README.md`](desktop/README.md#desktop-packages).

### Frontend development

The WebUI and Electron GUI use one React + TypeScript renderer from
`hhtools/web/frontend`; Electron loads the same page through its local FastAPI sidecar, so there is
no second GUI renderer. Video-to-motion is the first complete workflow in the new shell. Tailwind
CSS is wired into the build, and shadcn/ui primitives are copied into the project only when a real
view needs them. The Stage controls exist, while the new renderer is still being connected.

```bash
cd hhtools/web/frontend
npm install
npm run typecheck
npm test
npm run build
```

The production build is written to `hhtools/web/static`, which is served unchanged by both
FastAPI and Electron.

Web jobs are unlimited by default. To enable FIFO admission control on a shared or
memory-constrained GPU, set positive concurrency and an optional queue capacity:

```bash
uv run hhtools web --max-running-jobs 1 --max-queued-jobs 32
```

`0` means unlimited for both options; the queue setting only applies when running concurrency is
limited. Agent batches likewise have no configured item or total-frame cap by default. Positive
`--max-batch-items` and `--max-batch-total-frames` values opt into deployment-specific safeguards;
`0` restores unlimited mode. The same settings are available as `HHTOOLS_MAX_RUNNING_JOBS`,
`HHTOOLS_MAX_QUEUED_JOBS`, `HHTOOLS_MAX_BATCH_ITEMS`, and
`HHTOOLS_MAX_BATCH_TOTAL_FRAMES` (including in the Electron sidecar and MCP server).
They can also be edited under **Settings → Background-job scheduling** from local Web/Electron
or an SSH loopback tunnel; ordinary remote-browser sessions are shown read-only until authenticated
remote administration is implemented. Saving hot-applies the limits without restarting Python or Electron: lower running limits grandfather active jobs,
while higher limits immediately promote FIFO waiters. The backend persists the values in the
platform user-config directory; `HHTOOLS_WEB_SETTINGS_PATH` selects another file. Explicit CLI
or environment values remain startup overrides and will win again on the next launch.
The concurrency cap applies to scheduled Web jobs, not the optional Warp/Newton robot prewarm
thread, so it is admission control rather than a process-wide GPU concurrency guarantee.

| Panel | Flow |
|-------|------|
| **Video → Motion** | Upload one video → run official GVHMR → register the result in Motion Library |
| **Motion → Robot** | Load clip → select robot → calibrate (once) → retarget → download CSV/ZIP |
| **Robot → Robot** | Source robot + trajectory → target URDF → calibrate → retarget / batch ZIP |
| **Dataset analysis** | Drop a folder → analyze → explore tags & scatter → export subset |

### GVHMR video-to-motion

Install [GVHMR](https://github.com/zju3dv/GVHMR) separately using its upstream instructions. hhtools
does not bundle its source, official checkpoints, Python environment, or licensed body models in
the default DEB/EXE. An explicitly authorized local build can stage `SMPLX_NEUTRAL.npz`; source and
WebUI installs continue to use a local model directory.
The **Video → Motion** view runs inference with the official released weights and publishes the
generated `hmr4d_results.pt` to the Motion Library. Preprocessing caches stay outside the Library,
and uploaded video is normalized to the 30 FPS timeline expected by GVHMR. Custom checkpoints and
training are not exposed.

On Linux, use **Set up** in the desktop Video → Motion view to select the official checkout, that
installation's Python executable, and (when stored separately) the directory containing `smplx/`.
The paths are saved in desktop user data and passed only to the isolated GVHMR subprocess. The same
configuration can be supplied when launching from a terminal:

```bash
export HHTOOLS_GVHMR_ROOT=/path/to/GVHMR
export HHTOOLS_GVHMR_PYTHON=/path/to/gvhmr/environment/bin/python
export HHTOOLS_GVHMR_BODY_MODELS=/path/to/body_models
uv run hhtools web
```

The checkout must keep the upstream `inputs/checkpoints` layout, including the licensed
`inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`. The selected environment must provide the
GVHMR dependencies and CUDA, and `ffmpeg` must be on `PATH`. `~/GVHMR`, repository `.venv`/`venv`,
and common Conda environments named `gvhmr` are auto-discovered; explicit paths are more reliable.
`HHTOOLS_GVHMR_TIMEOUT_SECONDS` optionally changes the two-hour inference timeout.

On Windows, GVHMR remains an optional Docker-backed component. Set `HHTOOLS_GVHMR_ROOT` to the
official checkout and `HHTOOLS_GVHMR_IMAGE` to the prepared image;
`HHTOOLS_GVHMR_BODY_MODELS` can mount separately stored licensed assets. hhtools does not install or
download any of these resources.

An existing GVHMR result can still be dragged into **Motion** for import. Conversion requires a
locally licensed SMPL-family model; if it is outside hhtools' search paths, set
`HHTOOLS_BODY_MODELS` to that directory.

For a terminal-only workflow, convert a GVHMR output directory to hhtools' unified Motion format:

```bash
hhtools import run --dataset gvhmr --root /path/to/gvhmr/output --out /path/to/motions
```

Robot tuning: edit [`configs/robots/unitree_g1/`](configs/robots/unitree_g1/) or uploaded `~/.config/hhtools/robots/<name>/robot.yaml`; run `hhtools robot validate <name>`. Details in [framework.md](framework.md).

### CLI (batch / no Web UI)

Entry point: `uv run hhtools` (same package as the Web UI). Use this for large datasets (thousands of clips) instead of dragging files into the browser. Calibrate once in the Web UI (or place `retarget_calibration_<ref>.yaml` next to the URDF) before batch retarget.

| Command | Purpose |
|---------|---------|
| `hhtools doctor` | Check local Web, robot, retarget, MCP, body-model, and GVHMR readiness |
| `hhtools convert run` | BVH / GLB → unified NPZ |
| `hhtools import list` / `import run` | List adapters; import a dataset root → NPZ |
| `hhtools bodymodel check` / `setup` | SMPL-family weight paths / download hints |
| `hhtools robot list` / `info` / `schema` / `validate` / `scaffold` / `add` | Robot presets |
| `hhtools retarget run` | Newton IK → CSV (files or directory) |
| `hhtools retarget interaction-mesh run` | Interaction-mesh (terrain / objects) → CSV |
| `hhtools retarget interaction-mesh precompute-laplacian` | Precompute Laplacian targets (`.npz`) |
| `hhtools web` | HTML / three.js UI (default `127.0.0.1:8009`) |

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
uv run hhtools import run --dataset omnicontact \
  --root /path/to/OmniContact-Dataset -o /tmp/omnicontact_npz
```

**Robots**

```bash
# Install all six curated robots into ~/.config/hhtools/robots.
uv run python scripts/install_builtin_robots.py
# Install one preset only (repeat --only to select several).
uv run python scripts/install_builtin_robots.py --only g1_29dof

uv run hhtools robot list
uv run hhtools robot info g1_29dof --no-mjcf
uv run hhtools robot schema g1_29dof -o /tmp/g1_header.csv
uv run hhtools robot validate g1_29dof
uv run hhtools robot scaffold unitree_g1          # skip existing yaml
# uv run hhtools robot add /path/to/urdf_or_dir  # ingest into configs/robots/
```

The installer downloads pinned official archives (about 772 MiB total) and
keeps about 237 MiB of referenced files. Each preset retains its upstream
license and a checksummed `SOURCE.json`. `--replace` atomically replaces local
changes in the selected preset. Desktop bundles can point
`HHTOOLS_BUNDLED_ROBOT_DIR` at an audited Robot Library.

**Retarget (smoke with `--limit-frames`)**

```bash
# Newton IK (flat / AMASS-style NPZ)
uv run hhtools retarget run path/to/clip.npz \
  --robot unitree_g1__g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh (OMOMO / OmniContact / terrain clips)
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot unitree_g1__g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

**Large offline batches** (resumable, subprocess isolation; export matches Web CSV/sidecars, folders not zipped):

```bash
# mimic (flat mocap → Newton IK): amass | lafan | xsens_mocap (100STYLE) | glb | …
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset amass \
  --in /path/to/AMASS --out /path/to/AMASS_rp1 \
  --skip-existing --limit 5

# 100STYLE (Xsens MVN BVH → same adapter / calibration as xsens_mocap)
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset xsens_mocap \
  --in /path/to/100STYLE --out /path/to/100STYLE_rp1 \
  --skip-existing

# intermimic (human–object): omomo | omnicontact
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omomo \
  --in /path/to/OMOMO --out /path/to/OMOMO_rp1 \
  --skip-existing
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omnicontact \
  --in /path/to/OmniContact-Dataset --out /path/to/OmniContact_rp1 \
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

# MotionDecode (Unitree G1 CSV @ 120 Hz; files omit time / sample_rate)
python scripts/batch_r2r_retarget.py \
  --source-robot g1 --target-robot rp1 \
  --in /path/to/MotionDecode/samples --out /path/to/MotionDecode_rp1 \
  --source-fps 120 --skip-existing
```

Scene clips → `<out>/<clip>/<clip>.csv` + terrain/object sidecars (robot frame). Flat mimic → `<out>/…/<stem>.csv`. Use `--t-start` / `--t-end` (seconds on the retargeted timeline) to export a sub-clip; the Web single/batch export UI has the same option. Interaction-mesh needs `mujoco` + `osqp`; Newton needs the NVIDIA `newton` package. R2R needs a saved `r2r_calibration_<source>.yaml` beside the target URDF (Web calibrate once, or `--calibration` / `--init-zero-calibration`).

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
| mimic | [100STYLE](https://www.ianxmason.com/100style/) | [ACM](https://dl.acm.org/doi/10.1145/3522618) | [site](https://www.ianxmason.com/100style/) |
| mimic | Motion-X | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf) | [GitHub](https://github.com/IDEA-Research/Motion-X) |
| mimic | PHUMA | [arXiv](https://arxiv.org/abs/2510.26236) | [GitHub](https://github.com/DAVIAN-Robotics/PHUMA) |
| mimic | SOMA | [arXiv](https://arxiv.org/abs/2603.16858) | [Hugging Face](https://huggingface.co/datasets/bones-studio/seed) |
| intermimic | OMOMO | [arXiv](https://arxiv.org/abs/2309.16237) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_omomo) |
| intermimic | OmniContact-Dataset | [arXiv](https://arxiv.org/abs/2606.26201) | [Hugging Face](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) |
| meshmimic | holosoma | [arXiv](https://arxiv.org/abs/2509.26633) | [GitHub](https://github.com/amazon-far/holosoma) |
| meshmimic | PARC MS | [arXiv](https://arxiv.org/abs/2505.04002) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_parc_ms) |
| R2R | [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) | [site](https://chingmudata.github.io/MotionDecode/) | [Hugging Face](https://huggingface.co/datasets/CMRobot/MotionDecode) |

**100STYLE** is Xsens MVN BVH (60 fps stylized locomotion). Drop the unzipped tree under a folder named `100STYLE`, `xsens`, or `xsens_mocap` (for example `assets/motions/mimic/100STYLE/`) so the Web library picks it up. Calibrate the robot once with reference `xsens_mocap` — rest is the format T-pose, not a clip’s first frame. Single-file drops are auto-detected from joint names.

**OmniContact-Dataset** is optical-mocap human–object interaction (typically 90 Hz). Use the official [`raw_mocap/`](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) tree (`motion_actor.bvh` + object-pose CSV), not the already-retargeted G1 `npz/` files. Place the Hugging Face root (or just `raw_mocap/`) under a folder named `OmniContact-Dataset` — for example `assets/motions/intermimic/OmniContact-Dataset/`. Object meshes are picked up from a sibling `assets/` directory when present. Retarget with the interaction-mesh backend (`hhtools retarget interaction-mesh` / `scripts/batch_intermimic_retarget.py --dataset omnicontact`). The default calibration reference is the detected BVH dialect (`lafan_bvh` if unknown).

**MotionDecode** ([ChingMu](https://huggingface.co/datasets/CMRobot/MotionDecode)) ships **Unitree G1** retargeted CSVs under `samples/` (120 Hz; `root_pos_{xyz}(m)` + `root_rot` **wxyz** + `dof_*(rad)`). This is a **robot→robot** source, not a human-mocap adapter: use the Web **Robot → Robot** panel (source robot = `g1`) or `scripts/batch_r2r_retarget.py`. The files have no `time` / `# sample_rate`, so set **source FPS to 120** (Web “源轨迹 FPS”, or `--source-fps 120`); the default 50 Hz will play and retarget at the wrong speed. Nested taxonomy folders are scanned as R2R mimic clips. Please credit ChingMu when you use the data.

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
- Motion Library parameter clips still load as an animated skeleton when those weights are absent; the exact body mesh and joint regression require the locally licensed files.
- **More docs:** [framework.md](framework.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
