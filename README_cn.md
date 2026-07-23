# human-humanoid-tools（hhtools）

**让人形机器人在约 30 秒内完成跑酷 / 跳舞 / 交互动作的重映射**

**[项目主页](https://roboparty.github.io/human-humanoid-tools/)** · **[English README](README.md)**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Roboparty%2Fhuman--humanoid--tools-blue)](https://github.com/Roboparty/human-humanoid-tools)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue)](https://roboparty.github.io/human-humanoid-tools/)

| | |
| :---: | :---: |
| ![](assets/readme/demo-01.gif) | ![](assets/readme/demo-02.gif) |
| ![](assets/readme/demo-03.gif) | ![](assets/readme/demo-04.gif) |

---

欢迎提出任何建议和想法，可通过 Issue 或 Discussion 随时反馈。建议的新功能会在基础功能稳定之后再考虑加入。

---

## 亮点

- **快速重映射**：Web UI 或 **CLI**（`hhtools retarget` / `scripts/batch_*_retarget.py`）；**Newton IK** + **MPC-SQP** 交互网格。
- **多源人体数据**：BVH / GLB / SMPL 系；适配 AMASS、GVHMR、LAFAN、OMOMO、PHUMA、intermimic、meshmimic 等。
- **任意 URDF**：Web 上传任意其他机器人。拖入 URDF，拖入 mesh，自动识别，无需调参。
- **机器人→机器人（R2R）**：已有机器人 CSV/PKL 轨迹重映射到新 URDF。
- **数据集分析**：Web 端扫描、打标、聚类、子集推荐。

**环境：** Linux，Python 3.12+；预览 CPU 即可，重映射需 **NVIDIA GPU（CUDA 12）**。

---

## 快速开始

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
curl -LsSf https://astral.sh/uv/install.sh | sh   # 若未安装
uv sync --extra all
uv run hhtools web
```

浏览器打开 `http://127.0.0.1:8009`。

| 面板 | 流程 |
|------|------|
| **Motion → Robot** | 加载动作 → 选机器人 → 标定（首次）→ Retarget → 下载 CSV/ZIP |
| **Robot → Robot** | 源机器人 + 轨迹 → 目标 URDF → 标定 → 单条/批量导出 |
| **数据集可视化分析** | 拖入文件夹 → 分析 → 标签/散点探索 → 导出子集 |

参数调优：改 [`configs/robots/unitree_g1/`](configs/robots/unitree_g1/) 或 `~/.config/hhtools/robots/<名称>/robot.yaml`，运行 `hhtools robot validate <名称>`。原理见 [framework.md](framework.md)。

### CLI（批量 / 不走 Web）

入口：`uv run hhtools`（与 Web 同一套包）。上万条数据请用 CLI/脚本，不要往浏览器里拖。批量前请先在 Web 标定一次（或准备好 URDF 旁的 `retarget_calibration_<ref>.yaml`）。

| 命令 | 作用 |
|------|------|
| `hhtools convert run` | BVH / GLB → 统一 NPZ |
| `hhtools import list` / `import run` | 列出适配器；数据集根目录 → NPZ |
| `hhtools bodymodel check` / `setup` | SMPL 系权重路径 / 下载说明 |
| `hhtools robot list` / `info` / `schema` / `validate` / `scaffold` / `add` | 机器人预设 |
| `hhtools retarget run` | Newton IK → CSV（文件或目录） |
| `hhtools retarget interaction-mesh run` | Interaction-mesh（地形/物体）→ CSV |
| `hhtools retarget interaction-mesh precompute-laplacian` | 预计算 Laplacian 目标（`.npz`） |
| `hhtools web` | HTML / three.js UI（默认 `127.0.0.1:8009`） |
| `hhtools ui` | 旧版 Viser 查看器 |

**转换与导入**

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

**机器人**

```bash
uv run hhtools robot list
uv run hhtools robot info unitree_g1__g1_29dof --no-mjcf
uv run hhtools robot schema unitree_g1__g1_29dof -o /tmp/g1_header.csv
uv run hhtools robot validate unitree_g1__g1_29dof
uv run hhtools robot scaffold unitree_g1          # 已有 yaml 则跳过
# uv run hhtools robot add /path/to/urdf_or_dir  # 写入 configs/robots/
```

**重映射（可用 `--limit-frames` 冒烟）**

```bash
# Newton IK（平坦 / AMASS 类 NPZ）
uv run hhtools retarget run path/to/clip.npz \
  --robot unitree_g1__g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh（OMOMO / 带地形）
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot unitree_g1__g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

**大批量离线脚本**（可断点续跑、子进程隔离；导出内容与 Web 一致，场景 clip 保留文件夹不打 zip）：

```bash
# mimic（平坦 mocap → Newton IK）：amass | lafan | glb | …
python scripts/batch_mimic_retarget.py \
  --robot rp1 --dataset amass \
  --in /path/to/AMASS --out /path/to/AMASS_rp1 \
  --skip-existing --limit 5

# intermimic（人–物）：omomo
python scripts/batch_intermimic_retarget.py \
  --robot rp1 --dataset omomo \
  --in /path/to/OMOMO --out /path/to/OMOMO_rp1 \
  --skip-existing

# meshmimic（地形）：parc_ms | holosoma
python scripts/batch_meshmimic_retarget.py \
  --robot rp1 --dataset parc_ms \
  --in /path/to/parc_ms --out /path/to/parc_ms_rp1 \
  --skip-existing --failure-log failures.jsonl

# robot→robot（输入为已导出的源机轨迹树）
python scripts/batch_r2r_retarget.py \
  --source-robot rp1 --target-robot unitree_g1__g1_29dof \
  --in /path/to/rp1_exports --out /path/to/g1_from_rp1 \
  --profile auto --skip-existing
```

场景 clip → `<out>/<clip>/<clip>.csv` + 地形/物体 sidecar（机器人坐标系）。平坦 mimic → `<out>/…/<stem>.csv`。Interaction-mesh 需要 `mujoco` + `osqp`；Newton 需要 NVIDIA `newton` 包。R2R 需要目标机旁已有 `r2r_calibration_<source>.yaml`（先在 Web 标定，或 `--calibration` / `--init-zero-calibration`）。

### 调整 `robot.yaml`

路径：仓库内置机器人在 `configs/robots/<名称>/`；Web 上传的机器人在 `~/.config/hhtools/robots/<名称>/`。**改 yaml 后下次 Retarget 即生效，无需重启 Web**；仅升级 Python 包后需重启 `hhtools web`。

| 字段 | 作用 |
|------|------|
| `ik_map` | 标准人体关节 → URDF link。三自由度髋/肩应映射到**中间** link（多为 `*_roll_link`）。 |
| `weights` | IK 权重：`t_weight` 位置、`r_weight` 朝向。 |
| **`smooth_joint_filter_masks`** | **对 retarget 姿态影响很大的 IK 正则项**（与 pipeline 默认 `smooth_joint_filter_weight: 5.5` 配合）。按 link 名给 `[0, 1]` 系数，把关节往限位**中点**拉——**不是** `weights` 里的 tracking 权重。脚手架默认（如 `*_shoulder_roll_link: 1.0`）适合 G1/RP1 那种 roll 主要在 null space 的万向节；对**上传 URDF** 若手臂主要靠 **shoulder roll** 才能垂下，**`1.0` 会把手臂锁在张开位**，即使 `weights` 已调高也不跟踪。黄色骨架已下垂、机器人仍张臂时，**优先把 roll 降到 `0.1`–`0.3`**（要最大自由度可用 `0`），pitch/yaw 可保持中等以防抖动。 |
| `retarget.joint_scale_multipliers` | 可选。各 canonical 关节的**绝对**缩放覆盖（与标定 `derived.scales` 同单位），仅用于**手动**微调体型，无需重新标定。例如 `left_shoulder: 0.5` 收窄上半身。**不要**把某份标定的 `derived.scales` 整表贴进来（会串到其它数据集）。与**当前或任一** `retarget_calibration_*.yaml` 的 scales 相同（或仍是 scaffold 零位默认）则视为未修改并忽略。**肩**只影响横向 IK 与 shoulder roll，不改变竖直身高。 |
| `retarget.feet_stabilizer`、`apply_feet_stabilizer` | 脚底贴地、身体离地高度等；翻滚类动作可设 `apply_feet_stabilizer: false`。 |
| `retarget.references.<格式>` | 按动作格式覆盖（如 bundled `scaler_config`）。 |

```yaml
retarget:
  joint_scale_multipliers:
    left_shoulder: 0.5
    right_shoulder: 0.5
    left_elbow: 1.0
    # … 其余 ik_map 关节；与标定一致可省略
```

**`smooth_joint_filter_masks` 示例** — 若 mocap 手臂下垂、机器人仍 A 字张开，**先查此项**，不要只改 `weights`：

```yaml
smooth_joint_filter_masks:
  left_shoulder_pitch_link: 0.1
  left_shoulder_roll_link: 0.1   # 需要 roll 参与摆臂时不要用 1.0
  left_shoulder_yaw_link: 0.3
  right_shoulder_pitch_link: 0.1
  right_shoulder_roll_link: 0.1
  right_shoulder_yaw_link: 0.3
```

完整模板见 [`configs/robots/_template/robot.yaml`](configs/robots/_template/robot.yaml)。**重新上传 URDF** 会按 URDF 重新生成 `robot.yaml`（标定文件保留；手改的 `ik_map` / weights 可能被覆盖）。

**常见问题：** `git pull` 后请 `uv sync` 并重启 `uv run hhtools web`（勿用系统旧包）；硬刷新浏览器。Newton 批量失败会自动逐条回退；翻滚类动作请关闭「脚底贴地修正」。

---

## 演示动作（`assets/motions`）

仅含演示片段；完整数据请从上游下载。本工具只提供格式适配，**不重新分发**数据集。

| 模式 | 数据集 | 论文 | 下载 |
|------|--------|------|------|
| mimic | AMASS | [arXiv](https://arxiv.org/abs/1904.03278) | [官网](https://amass.is.tue.mpg.de/) |
| mimic | GVHMR | [arXiv](https://arxiv.org/abs/2409.06662) | [GitHub](https://github.com/zju3dv/GVHMR) |
| mimic | LAFAN1 | [arXiv](https://arxiv.org/abs/2102.04942) | [GitHub](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) |
| mimic | Motion-X | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf) | [GitHub](https://github.com/IDEA-Research/Motion-X) |
| mimic | PHUMA | [arXiv](https://arxiv.org/abs/2510.26236) | [GitHub](https://github.com/DAVIAN-Robotics/PHUMA) |
| mimic | SOMA | [arXiv](https://arxiv.org/abs/2603.16858) | [Hugging Face](https://huggingface.co/datasets/bones-studio/seed) |
| intermimic | OMOMO | [arXiv](https://arxiv.org/abs/2309.16237) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_omomo) |
| meshmimic | holosoma | [arXiv](https://arxiv.org/abs/2509.26633) | [GitHub](https://github.com/amazon-far/holosoma) |
| meshmimic | PARC MS | [arXiv](https://arxiv.org/abs/2505.04002) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_parc_ms) |

---

## 引用

若在论文或项目中使用 **human-humanoid-tools**，请引用本仓库：

```bibtex
@software{human_humanoid_tools2026,
  title        = {human-humanoid-tools (hhtools): humanoid motion retargeting and dataset analysis},
  author       = {jaggerShen and hhtools contributors},
  year         = {2026},
  url          = {https://github.com/Roboparty/human-humanoid-tools},
  license      = {Apache-2.0}
}
```

**链接：** [GitHub 仓库](https://github.com/Roboparty/human-humanoid-tools) · [Issues](https://github.com/Roboparty/human-humanoid-tools/issues) · [LICENSE](LICENSE)

使用内置数据集适配器时，请同时引用对应 **上游数据集与算法**（见上表及 [NOTICE](NOTICE)，如 SOMA-Retargeter、holosoma）。

---

## 许可证

- **代码：** [Apache-2.0](LICENSE) · 第三方：[NOTICE](NOTICE)
- **SMPL 系权重：** 不随仓库分发，需自行从 MPI 下载并放入 `configs/body_models/` — 见 [configs/body_models/README.md](configs/body_models/README.md)
- **更多文档：** [framework.md](framework.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
