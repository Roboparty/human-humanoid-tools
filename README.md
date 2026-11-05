<div align="center">

<img src="hhtools/web/frontend/public/hhtools-robot.svg" alt="HHTools logo" width="88">

# Human-Humanoid Tools

**Retarget parkour, dance, and interaction clips onto any humanoid in ~30 seconds.**

[![Latest release](https://img.shields.io/github/v/release/Roboparty/human-humanoid-tools?include_prereleases&label=release&color=3f78c7)](https://github.com/Roboparty/human-humanoid-tools/releases/latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3f78c7.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-3f78c7.svg)](LICENSE)

[**Download**](#download) · [**Run from source**](#run-from-source) · [**Docs**](#documentation) · [**Project page**](https://roboparty.github.io/human-humanoid-tools/) · [**中文说明**](README_cn.md)

</div>

<p align="center">
  <img src="assets/readme/demo-01.gif" width="49%" alt="Retargeting demo 1">
  <img src="assets/readme/demo-02.gif" width="49%" alt="Retargeting demo 2">
  <img src="assets/readme/demo-03.gif" width="49%" alt="Retargeting demo 3">
  <img src="assets/readme/demo-04.gif" width="49%" alt="Retargeting demo 4">
</p>

## What you can do

| Workflow | How it works |
|----------|--------------|
| **Motion&nbsp;→&nbsp;Robot** | Load a clip → pick a robot → calibrate once → retarget → export. Newton IK for flat motion; MPC-SQP interaction mesh for terrain and objects. |
| **Robot&nbsp;→&nbsp;Robot** | Retarget existing robot trajectories (CSV, PKL, IsaacLab NPZ) onto a new URDF. |
| **Video&nbsp;→&nbsp;Motion** | Run official [GVHMR](docs/gvhmr.md) on a video and add the result to your Motion Library (optional, needs CUDA). |
| **Dataset&nbsp;analysis** | Point at a folder → scan, tag, embed, and cluster → export a subset. |

- **Bring any robot.** Drag in a URDF and its meshes; joints are detected automatically.
- **Bring your data.** BVH, GLB, and the SMPL family, with adapters for AMASS, LAFAN, 100STYLE, OMOMO, OmniContact, PHUMA, and [more](docs/datasets.md).
- **Trust the result.** Diagnostics compare the robot with the IK targets it tracked and grade the error relative to robot height.
- **Automate it.** Batch from the [CLI](docs/cli.md), or drive jobs from an agent over [JSON CLI or MCP](docs/agent.md).

## Download

The desktop app is the quickest way to start. It includes 30 motions and seven robots: ROBOTO 01, Unitree G1, ROBOTO_ORIGIN, AgiBot X2, Asimov 1, Fourier GR-2, and Berkeley Humanoid Lite.

| Platform | Download (0.1.6) | Good to know |
|----------|------------------|--------------|
| **Windows** x64 | [EXE installer](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe) | Ready to run, no Python needed. Self-signed, so SmartScreen may warn. |
| **Linux** amd64 | [Debian package](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb) | First launch sets up a private Python environment (needs internet). |
| **macOS** 12+, Apple Silicon | [DMG installer](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg) | Same first-launch setup. Not notarized; allow it under Privacy & Security. Retargets on CPU, no Video → Motion. |

See the [installation guide](docs/install.md) for step-by-step instructions and checksums, or browse [all releases](https://github.com/Roboparty/human-humanoid-tools/releases).

## Run from source

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
uv sync --locked --extra all
uv run python scripts/install_builtin_robots.py   # add --only rp1 for a single robot
uv run hhtools web                                # then open http://127.0.0.1:8009
```

Linux and Windows use an NVIDIA GPU (CUDA 12) when available. Smaller installs are covered in the [installation guide](docs/install.md#from-source).

### Ways to run

All of them share the same motion, robot, and retargeting core.

| Mode | Best for | Launch |
|------|----------|--------|
| **Desktop&nbsp;app** | Getting started, no terminal needed | Application menu or `hhtools-desktop` |
| **WebUI** | Interactive work on a workstation or server | `uv run hhtools web` |
| **CLI** | Large datasets, SSH, and scripts | `uv run hhtools ...` |
| **Agent** | Scripted H2R, scene-free R2R, and batch jobs (a subset of the WebUI) | [`hhtools agent` / `hhtools-mcp`](docs/agent.md) |

## Documentation

| Guide | What's inside |
|-------|---------------|
| [Installation](docs/install.md) | Desktop setup per platform, checksums, source install options |
| [CLI and batch jobs](docs/cli.md) | Commands, converting and importing data, offline batch retargeting |
| [Robot tuning](docs/robot-tuning.md) | Adding robots, `robot.yaml` fields, fixing common pose problems |
| [Datasets](docs/datasets.md) | Supported datasets, where to get them, per-dataset notes |
| [Video → Motion](docs/gvhmr.md) | Setting up GVHMR on Linux and Windows |
| [Agent and MCP](docs/agent.md) | Installation, scope, and the smoke-first workflow |
| [Job scheduling](docs/job-scheduling.md) | Limiting concurrent jobs on shared or memory-constrained GPUs |
| [Desktop packaging](desktop/README.md) | Building the desktop packages |
| [Architecture](framework.md) · [Contributing](CONTRIBUTING.md) | How the code is organized and how to work on it |

## Feedback

Suggestions and ideas are welcome, so please [open an issue](https://github.com/Roboparty/human-humanoid-tools/issues). New feature requests will be considered once the core functionality is stable.

## Citation

If you use HHTools in research or products, please cite this repository, along with the upstream datasets and solvers you rely on (see [Datasets](docs/datasets.md) and [NOTICE](NOTICE)).

```bibtex
@software{human_humanoid_tools2026,
  title        = {human-humanoid-tools (hhtools): humanoid motion retargeting and dataset analysis},
  author       = {jaggerShen and hhtools contributors},
  year         = {2026},
  url          = {https://github.com/Roboparty/human-humanoid-tools},
  license      = {Apache-2.0}
}
```

## License

Code is [Apache-2.0](LICENSE); third-party notices are in [NOTICE](NOTICE). Datasets are not redistributed. SMPL, SMPL-H, and SMPL-X weights are not included: register with MPI and follow the [body model guide](configs/body_models/README.md). Without them, parameter clips still load as an animated skeleton, but the exact body mesh and joint regression are unavailable.
