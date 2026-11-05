# Video → Motion (GVHMR)

**[中文](./gvhmr_cn.md)**

The **Video → Motion** view runs inference with the official released [GVHMR](https://github.com/zju3dv/GVHMR) weights and publishes the generated `hmr4d_results.pt` to the Motion Library. Preprocessing caches stay outside the Library, and uploaded video is normalized to the 30 FPS timeline GVHMR expects. Custom checkpoints and training are not exposed.

Install GVHMR separately using its upstream instructions. HHTools does not bundle its source, official checkpoints, Python environment, or licensed body models in the default DEB or EXE. An explicitly authorized local build can stage `SMPLX_NEUTRAL.npz`; source and WebUI installs keep using a local model directory.

This workflow is not available in the macOS desktop app.

## Linux

Use **Set up** in the desktop Video → Motion view to select the official checkout, that installation's Python executable, and (when stored separately) the directory containing `smplx/`. The paths are saved in desktop user data and passed only to the isolated GVHMR subprocess.

When launching from a terminal, supply the same configuration through the environment:

```bash
export HHTOOLS_GVHMR_ROOT=/path/to/GVHMR
export HHTOOLS_GVHMR_PYTHON=/path/to/gvhmr/environment/bin/python
export HHTOOLS_GVHMR_BODY_MODELS=/path/to/body_models
uv run hhtools web
```

Requirements:

- The checkout must keep the upstream `inputs/checkpoints` layout, including the licensed `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`.
- The selected environment must provide the GVHMR dependencies and CUDA.
- `ffmpeg` must be on `PATH`.

`~/GVHMR`, repository `.venv` / `venv`, and common Conda environments named `gvhmr` are auto-discovered, but explicit paths are more reliable. `HHTOOLS_GVHMR_TIMEOUT_SECONDS` optionally changes the two-hour inference timeout.

## Windows

GVHMR is an optional Docker-backed component. Set `HHTOOLS_GVHMR_ROOT` to the official checkout and `HHTOOLS_GVHMR_IMAGE` to the prepared image. `HHTOOLS_GVHMR_BODY_MODELS` can mount separately stored licensed assets. HHTools does not install or download any of these resources.

## Using existing GVHMR results

An existing GVHMR result can be dragged into **Motion** for import. Conversion requires a locally licensed SMPL-family model; if it is outside HHTools' search paths, set `HHTOOLS_BODY_MODELS` to that directory.

For a terminal-only workflow, convert a GVHMR output directory to the unified Motion format:

```bash
hhtools import run --dataset gvhmr --root /path/to/gvhmr/output --out /path/to/motions
```
