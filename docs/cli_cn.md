# CLI 与批处理

**[English](cli.md)**

入口是 `uv run hhtools`，与 WebUI 是同一套包。处理大规模数据（上千条动作）时请使用 CLI，而不要往浏览器里拖文件。

**批量重映射前，** 请先在 WebUI 中标定一次，或在 URDF 旁放好 `retarget_calibration_<ref>.yaml`。

## 命令

| 命令 | 作用 |
|------|------|
| `hhtools doctor` | 检查本机 Web、机器人、重映射、MCP、Body Model 与 GVHMR 就绪状态 |
| `hhtools convert run` | BVH / GLB → 统一 NPZ |
| `hhtools import list` / `import run` | 列出适配器；数据集根目录 → NPZ |
| `hhtools bodymodel check` / `setup` | SMPL 系权重路径 / 下载说明 |
| `hhtools robot list` / `info` / `schema` / `validate` / `scaffold` / `add` | 机器人预设 |
| `hhtools retarget run` | Newton IK → CSV（文件或目录） |
| `hhtools retarget interaction-mesh run` | Interaction-mesh（地形 / 物体）→ CSV |
| `hhtools retarget interaction-mesh precompute-laplacian` | 预计算 Laplacian 目标（`.npz`） |
| `hhtools web` | HTML / three.js UI（默认 `127.0.0.1:8009`） |
| `hhtools agent` | 供自动化使用的严格 JSON 接口，见 [Agent 与 MCP](agent.md)（英文） |

## 转换与导入

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

## 机器人

先按 [内置机器人](install_cn.md#内置机器人) 安装预设，然后：

```bash
uv run hhtools robot list
uv run hhtools robot info rp1 --no-mjcf
uv run hhtools robot schema rp1 -o /tmp/rp1_header.csv
uv run hhtools robot validate rp1
uv run hhtools robot scaffold g1_29dof            # 已有 yaml 则跳过
# uv run hhtools robot add /path/to/urdf_or_dir  # 写入 configs/robots/
```

7 款预设为 ROBOTO 01（`rp1`，28 自由度）、Unitree G1（`g1_29dof`）、ROBOTO_ORIGIN（`roboto_origin`）、AgiBot X2（`agibot_x2_ultra`）、Asimov 1（`asimov_1`）、Fourier GR-2（`fourier_gr2`）和 Berkeley Humanoid Lite（`berkeley_humanoid_lite`）。

## 重映射

用 `--limit-frames` 快速冒烟测试：

```bash
# Newton IK（平坦 / AMASS 类 NPZ）
uv run hhtools retarget run path/to/clip.npz \
  --robot g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh（OMOMO / OmniContact / 带地形）
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

## 大批量离线任务

批处理可断点续跑，并以子进程隔离运行。导出内容与 WebUI 的 CSV 和 sidecar 一致，场景 clip 保留文件夹，不打包为 zip。

```bash
# mimic（平坦 mocap → Newton IK）：amass | lafan | xsens_mocap（100STYLE）| glb | …
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset amass \
  --in /path/to/AMASS --out /path/to/AMASS_rp1 \
  --skip-existing --limit 5

# 100STYLE（Xsens MVN BVH，与 xsens_mocap 同一适配器 / 标定）
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset xsens_mocap \
  --in /path/to/100STYLE --out /path/to/100STYLE_rp1 \
  --skip-existing

# intermimic（人–物）：omomo | omnicontact
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omomo \
  --in /path/to/OMOMO --out /path/to/OMOMO_rp1 \
  --skip-existing
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omnicontact \
  --in /path/to/OmniContact-Dataset --out /path/to/OmniContact_rp1 \
  --skip-existing

# meshmimic（地形）：parc_ms | holosoma
python scripts/batch_meshmimic_retarget.py \
  --robot rp1 --dataset parc_ms \
  --in /path/to/parc_ms --out /path/to/parc_ms_rp1 \
  --skip-existing --failure-log failures.jsonl

# robot→robot（输入为已导出的源机器人轨迹）
python scripts/batch_r2r_retarget.py \
  --source-robot rp1 --target-robot g1_29dof \
  --in /path/to/rp1_exports --out /path/to/g1_from_rp1 \
  --profile auto --skip-existing

# MotionDecode（Unitree G1 CSV，120 Hz；文件无 time / sample_rate）
python scripts/batch_r2r_retarget.py \
  --source-robot g1_29dof --target-robot rp1 \
  --in /path/to/MotionDecode/samples --out /path/to/MotionDecode_rp1 \
  --source-fps 120 --skip-existing
```

**输出结构**

- 场景 clip：`<out>/<clip>/<clip>.csv`，以及地形和物体 sidecar（机器人坐标系）。
- 平坦 mimic：`<out>/…/<stem>.csv`。

**选项与依赖**

- `--t-start` / `--t-end`（秒，相对重映射后的时间轴）只导出一段。WebUI 的单条和批量导出也有相同选项。
- Interaction-mesh 需要 `mujoco` 和 `osqp`；Newton 需要 NVIDIA `newton` 包。
- GPU 批量 IK 因共享内存不足失败时，会自动改为逐条 IK。
- R2R 需要目标机器人 URDF 旁已有 `r2r_calibration_<source>.yaml`（先在 WebUI 中标定一次，或使用 `--calibration` / `--init-zero-calibration`）。
