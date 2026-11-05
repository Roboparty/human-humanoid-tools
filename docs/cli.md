# CLI and batch jobs

**[中文](./cli_cn.md)**

The entry point is `uv run hhtools`, the same package as the WebUI. Use it for large datasets (thousands of clips) instead of dragging files into the browser.

**Before batch retargeting,** calibrate once in the WebUI, or place `retarget_calibration_<ref>.yaml` next to the URDF.

## Commands

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
| `hhtools agent` | Strict JSON interface for automation; see [Agent and MCP](agent.md) |

## Convert and import

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

## Robots

Install presets as described in [Built-in robots](install.md#built-in-robots), then:

```bash
uv run hhtools robot list
uv run hhtools robot info rp1 --no-mjcf
uv run hhtools robot schema rp1 -o /tmp/rp1_header.csv
uv run hhtools robot validate rp1
uv run hhtools robot scaffold g1_29dof            # skip existing yaml
# uv run hhtools robot add /path/to/urdf_or_dir  # ingest into configs/robots/
```

The seven presets are ROBOTO 01 (`rp1`, 28 DoF), Unitree G1 (`g1_29dof`), ROBOTO_ORIGIN (`roboto_origin`), AgiBot X2 (`agibot_x2_ultra`), Asimov 1 (`asimov_1`), Fourier GR-2 (`fourier_gr2`), and Berkeley Humanoid Lite (`berkeley_humanoid_lite`).

## Retarget

Use `--limit-frames` for a quick smoke test:

```bash
# Newton IK (flat / AMASS-style NPZ)
uv run hhtools retarget run path/to/clip.npz \
  --robot g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh (OMOMO / OmniContact / terrain clips)
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

## Large offline batches

Batches are resumable and run with subprocess isolation. Export matches the WebUI's CSV and sidecars, and folders are not zipped.

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
  --source-robot rp1 --target-robot g1_29dof \
  --in /path/to/rp1_exports --out /path/to/g1_from_rp1 \
  --profile auto --skip-existing

# MotionDecode (Unitree G1 CSV @ 120 Hz; files omit time / sample_rate)
python scripts/batch_r2r_retarget.py \
  --source-robot g1_29dof --target-robot rp1 \
  --in /path/to/MotionDecode/samples --out /path/to/MotionDecode_rp1 \
  --source-fps 120 --skip-existing
```

**Output layout**

- Scene clips: `<out>/<clip>/<clip>.csv` plus terrain and object sidecars (robot frame).
- Flat mimic: `<out>/…/<stem>.csv`.

**Options and dependencies**

- `--t-start` / `--t-end` (seconds on the retargeted timeline) export a sub-clip. The WebUI's single and batch export have the same option.
- Interaction-mesh needs `mujoco` and `osqp`; Newton needs the NVIDIA `newton` package.
- If GPU batch IK runs out of shared memory, the batch falls back to retargeting clips one at a time.
- R2R needs a saved `r2r_calibration_<source>.yaml` beside the target URDF (calibrate once in the WebUI, or use `--calibration` / `--init-zero-calibration`).
