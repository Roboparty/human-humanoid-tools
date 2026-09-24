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
- **多源人体数据**：BVH / GLB / SMPL 系；适配 AMASS、GVHMR、LAFAN、100STYLE、OMOMO、OmniContact、PHUMA、intermimic、meshmimic 等。
- **任意 URDF**：Web 上传任意其他机器人。拖入 URDF，拖入 mesh，自动识别，无需调参。
- **机器人→机器人（R2R）**：已有机器人 CSV/PKL 轨迹重映射到新 URDF，含 [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) 的 G1 CSV。
- **数据集分析**：Web 端扫描、打标、聚类、子集推荐。

**环境：** Python 3.12+；Linux / Windows 支持 **NVIDIA GPU（CUDA 12）** 加速，
Apple Silicon macOS 桌面版使用 CPU 重定向。桌面打包要求见下方说明。
视频转动作还需要一套单独安装、可使用 CUDA 的 GVHMR 环境。

---

## 安装与启动

hhtools 有三种交互式运行方式，另提供一个 Agent 自动化接口。它们共享同一套动作、机器人与
重映射核心，但安装和启动入口彼此独立：

| 方式 | 适用场景 | 启动入口 |
|------|----------|----------|
| **终端（CLI/TUI 工作流）** | 批处理、服务器、SSH 与自动化 | `uv run hhtools ...` |
| **WebUI** | 浏览器中的可视化与交互工作流 | `uv run hhtools web` |
| **桌面 GUI** | Windows 独立应用、Linux / macOS 首次启动安装 | 应用菜单、Mac 应用程序或 `hhtools-desktop` |
| **Agent（JSON CLI / MCP）** | 带版本契约的 H2R、无场景 R2R 与可扩展 Batch 自动化 | [`hhtools agent` / `hhtools-mcp`](docs/agent.md) |

### 推荐：使用 uv 安装源码环境

先按照 [`uv` 官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv，再克隆
仓库。HHTools 支持 Python 3.12 或更高版本；uv 会选择兼容解释器，也可以在本机缺少时安装一个：

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
# 仅在本机没有兼容 Python 时需要：
uv python install 3.12
```

需要格式、机器人、重映射、WebUI 与 Agent/MCP 全部功能时，使用项目维护的 `all` extra：

```bash
uv sync --locked --extra all
uv run hhtools --help
uv run hhtools web
```

浏览器打开 `http://127.0.0.1:8009`。GVHMR 与受单独许可约束的 SMPL 系模型权重仍由用户
独立准备；`all` 安装的是对应 Python 集成库，不包含这些模型文件。

如果希望环境更小，可以只安装实际工作流。核心终端命令不需要 extra：

```bash
uv sync --locked
uv run hhtools --help
```

需要 WebUI 预览和 Newton 重映射时：

```bash
uv sync --locked --extra web --extra retarget
uv run hhtools web
```

如果只需要预览，可省略 `--extra retarget`。缺少必需包时，启动程序会列出缺失包及准确的
修复命令，不再直接显示 Python import traceback。

### Agent 与 MCP

HHTools 提供供脚本使用的严格 JSON CLI，以及供兼容 Agent 使用的本机 stdio MCP server。
当前 Agent 接口覆盖经过 preflight 的安全 H2R、无场景 R2R、可扩展 H2R/R2R Batch 任务、
内容寻址的标定辅助、供 GPT 视觉检查的前/侧视预览，以及验证后的静默保存和产物导出，
尚未覆盖 WebUI 的全部功能。安装方式、能力边界、smoke-first 流程、
运行时目录所有权和仓库自带的 Codex 项目配置见 [Agent interfaces](docs/agent.md)。

### 桌面 GUI

按平台下载 **0.1.0 预览版**桌面安装包：

| 平台 | 下载 | 架构 |
|------|------|------|
| Windows | [EXE 安装程序](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-x64-setup.exe) | x64 |
| Linux | [Debian 安装包](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-amd64.deb) | amd64 |
| macOS 12+ | [DMG 安装包](https://github.com/Eleanor1018/human-humanoid-tools/releases/download/v0.1.0%28beta%29/hhtools-0.1.0-mac-arm64.dmg) | Apple Silicon（arm64） |

Debian 包包含 Electron、精选内置动作与机器人。首次启动页使用同一 deb 内置的 HHTools wheel、
锁定依赖清单、uv 配置和 uv 二进制安装独立运行环境；推荐的当前用户安装不需要管理员密码，
所有用户安装则使用操作系统认证窗口：

```bash
sudo apt install ./hhtools-0.1.0-amd64.deb
hhtools-desktop
```

Windows 安装程序会直接打包 Python runtime 与应用源码，不需要目标机器另行准备 checkout 或
系统 Python。

**macOS 安装步骤：**

1. 下载并打开上方 DMG，将 **Human-Humanoid Tools** 拖入 **Applications（应用程序）**。
2. 从“应用程序”启动。此预览版使用 ad-hoc 签名，尚未经过 Apple 公证；若被 macOS 阻止，
   请在 **系统设置 → 隐私与安全性 → 仍要打开** 中允许启动此应用。
3. 按首次启动页提示完成安装，下载独立 Python 3.12 环境期间请保持联网。无需预装 Python、
   源码仓库或 Homebrew；运行环境仅安装到当前用户目录，不需要管理员密码。

macOS 包与其他桌面版一样，包含 30 个内置动作和 6 款机器人。
Warp 重定向使用 CPU；不支持 Intel Mac 或本机 GVHMR 的 NVIDIA GPU 流程。

三端都不默认打包 GVHMR 和受单独许可约束的 SMPL 系权重。构建说明见
[`desktop/README.md`](desktop/README.md#desktop-packages)。

### 前端开发

WebUI 与 Electron GUI 共用 `hhtools/web/frontend` 中同一套 React + TypeScript renderer；
Electron 通过本机 FastAPI sidecar 加载同一个页面，不维护第二套 GUI renderer。视频转动作是
新壳层中第一个完整工作流。Tailwind CSS 已接入构建；shadcn/ui 基础组件只在真实页面需要时
逐个复制进项目。舞台控制已经存在，新 renderer 仍在接入中。

```bash
cd hhtools/web/frontend
npm install
npm run typecheck
npm test
npm run build
```

生产构建写入 `hhtools/web/static`，FastAPI 与 Electron 原样复用这份产物。

Web 后台任务默认不限制并发。共享服务器或显存紧张时，可以显式启用 FIFO 调度：

```bash
uv run hhtools web --max-running-jobs 1 --max-queued-jobs 32
```

两个参数的 `0` 都表示不限；只有运行并发为正数时，等待队列设置才生效。也可以使用
Agent Batch 的条目数和总帧数默认也不设上限；只有把 `--max-batch-items` 或
`--max-batch-total-frames` 设为正数时才启用部署侧保护，设回 `0` 即恢复不限。对应环境变量为
`HHTOOLS_MAX_RUNNING_JOBS`、`HHTOOLS_MAX_QUEUED_JOBS`、`HHTOOLS_MAX_BATCH_ITEMS` 和
`HHTOOLS_MAX_BATCH_TOTAL_FRAMES`，Electron sidecar 与 MCP server 同样支持。
也可以从本机 Web/Electron 或 SSH 本地回环隧道，在 **设置 → 后台任务调度** 中直接修改；
在未实现远程管理鉴权前，普通远程浏览器会显示为只读。保存会热更新调度器，无需重启 Python 或
Electron：降低并发不会中断正在运行的任务，提高上限会立即按 FIFO 补跑等待任务。后端会将
配置写入平台用户配置目录，也可用 `HHTOOLS_WEB_SETTINGS_PATH` 指定文件。显式 CLI/环境变量
仍是启动覆盖项，只要保留这些覆盖项，下次启动时就会再次覆盖 GUI 保存值。
并发上限只约束调度器管理的 Web Job，不包含选择机器人时可选的 Warp/Newton 预热线程，
因此它是任务准入控制，并非整个进程的严格 GPU 并发上限。

| 面板 | 流程 |
|------|------|
| **视频 → 动作** | 上传单个视频 → 使用 GVHMR 官方权重推理 → 登记到 Motion Library |
| **Motion → Robot** | 加载动作 → 选机器人 → 标定（首次）→ Retarget → 下载 CSV/ZIP |
| **Robot → Robot** | 源机器人 + 轨迹 → 目标 URDF → 标定 → 单条/批量导出 |
| **数据集可视化分析** | 拖入文件夹 → 分析 → 标签/散点探索 → 导出子集 |

### GVHMR 视频转动作

请按照 [GVHMR 上游说明](https://github.com/zju3dv/GVHMR)单独安装。hhtools 不捆绑其源码、
官方 checkpoint 或 Python 环境；本机 desktop 构建可以带入 `SMPLX_NEUTRAL.npz`，源码与 WebUI
仍使用本机模型目录。**视频 → 动作** 只使用官方发布
权重完成推理，并将生成的 `hmr4d_results.pt` 登记到 Motion Library；不提供自定义权重或训练入口。

Linux 端会用独立子进程直接启动已经安装好的 GVHMR Python 环境：

```bash
export HHTOOLS_GVHMR_ROOT=/path/to/GVHMR
export HHTOOLS_GVHMR_PYTHON=/path/to/gvhmr/environment/bin/python
uv run hhtools web
```

checkout 需要保留上游的 `inputs/checkpoints` 布局，其中包括已获得授权的
`inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`。所选环境必须具备 GVHMR 依赖与
CUDA，`ffmpeg` 必须位于 `PATH`。程序会尝试发现 `~/GVHMR`、仓库内 `.venv`/`venv` 以及常见
位置中名为 `gvhmr` 的 Conda 环境，但显式路径最可靠。可用 `HHTOOLS_GVHMR_TIMEOUT_SECONDS`
调整默认两小时的推理超时。

Windows 端仍是可选的 Docker 组件。用 `HHTOOLS_GVHMR_ROOT` 指向官方 checkout，
`HHTOOLS_GVHMR_IMAGE` 指向已准备好的镜像；如授权模型单独存放，可用
`HHTOOLS_GVHMR_BODY_MODELS` 挂载。hhtools 不会安装或下载这些资源。

已有的 GVHMR 结果仍可拖入 **Motion** 导入。转换过程需要本地已授权的 SMPL 系人体模型；
如果它不在 hhtools 默认搜索路径中，请用 `HHTOOLS_BODY_MODELS` 指向该目录。

纯命令行用户也可直接将 GVHMR 输出目录转成 hhtools 统一 Motion 格式：

```bash
hhtools import run --dataset gvhmr --root /path/to/gvhmr/output --out /path/to/motions
```

参数调优：改 [`configs/robots/unitree_g1/`](configs/robots/unitree_g1/) 或 `~/.config/hhtools/robots/<名称>/robot.yaml`，运行 `hhtools robot validate <名称>`。原理见 [framework.md](framework.md)。

### CLI（批量 / 不走 Web）

入口：`uv run hhtools`（与 Web 同一套包）。上万条数据请用 CLI/脚本，不要往浏览器里拖。批量前请先在 Web 标定一次（或准备好 URDF 旁的 `retarget_calibration_<ref>.yaml`）。

| 命令 | 作用 |
|------|------|
| `hhtools doctor` | 检查本机 Web、机器人、重映射、MCP、Body Model 与 GVHMR 就绪状态 |
| `hhtools convert run` | BVH / GLB → 统一 NPZ |
| `hhtools import list` / `import run` | 列出适配器；数据集根目录 → NPZ |
| `hhtools bodymodel check` / `setup` | SMPL 系权重路径 / 下载说明 |
| `hhtools robot list` / `info` / `schema` / `validate` / `scaffold` / `add` | 机器人预设 |
| `hhtools retarget run` | Newton IK → CSV（文件或目录） |
| `hhtools retarget interaction-mesh run` | Interaction-mesh（地形/物体）→ CSV |
| `hhtools retarget interaction-mesh precompute-laplacian` | 预计算 Laplacian 目标（`.npz`） |
| `hhtools web` | HTML / three.js UI（默认 `127.0.0.1:8009`） |

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
uv run hhtools import run --dataset omnicontact \
  --root /path/to/OmniContact-Dataset -o /tmp/omnicontact_npz
```

**机器人**

```bash
# 将六个精选机器人安装到 ~/.config/hhtools/robots。
uv run python scripts/install_builtin_robots.py
# 也可以只安装一个；重复 --only 可选择多个。
uv run python scripts/install_builtin_robots.py --only g1_29dof

uv run hhtools robot list
uv run hhtools robot info g1_29dof --no-mjcf
uv run hhtools robot schema g1_29dof -o /tmp/g1_header.csv
uv run hhtools robot validate g1_29dof
uv run hhtools robot scaffold unitree_g1          # 已有 yaml 则跳过
# uv run hhtools robot add /path/to/urdf_or_dir  # 写入 configs/robots/
```

安装器下载固定提交的官方归档（合计约 772 MiB），只保留约 237 MiB 的引用
文件；每个预设都会保留上游许可证和带校验和的 `SOURCE.json`。`--replace`
会原子替换所选预设的本地修改。桌面包可用 `HHTOOLS_BUNDLED_ROBOT_DIR`
指向审核后的 Robot Library。

**重映射（可用 `--limit-frames` 冒烟）**

```bash
# Newton IK（平坦 / AMASS 类 NPZ）
uv run hhtools retarget run path/to/clip.npz \
  --robot unitree_g1__g1_29dof -o /tmp/out.csv \
  --calibration-reference smpl --limit-frames 30

# Interaction-mesh（OMOMO / OmniContact / 带地形）
uv run hhtools retarget interaction-mesh run path/to/clip.pkl \
  --robot unitree_g1__g1_29dof -o /tmp/out_im.csv \
  --calibration-reference smpl --limit-frames 30
```

**大批量离线脚本**（可断点续跑、子进程隔离；导出内容与 Web 一致，场景 clip 保留文件夹不打 zip）：

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

# robot→robot（输入为已导出的源机轨迹树）
python scripts/batch_r2r_retarget.py \
  --source-robot rp1 --target-robot unitree_g1__g1_29dof \
  --in /path/to/rp1_exports --out /path/to/g1_from_rp1 \
  --profile auto --skip-existing

# MotionDecode（Unitree G1 CSV，120 Hz；文件无 time / sample_rate）
python scripts/batch_r2r_retarget.py \
  --source-robot g1 --target-robot rp1 \
  --in /path/to/MotionDecode/samples --out /path/to/MotionDecode_rp1 \
  --source-fps 120 --skip-existing
```

场景 clip → `<out>/<clip>/<clip>.csv` + 地形/物体 sidecar（机器人坐标系）。平坦 mimic → `<out>/…/<stem>.csv`。可用 `--t-start` / `--t-end`（秒，相对 Retarget 时间轴）只导出一段；Web 单条/批量导出也有对应选项。Interaction-mesh 需要 `mujoco` + `osqp`；Newton 需要 NVIDIA `newton` 包。R2R 需要目标机旁已有 `r2r_calibration_<source>.yaml`（先在 Web 标定，或 `--calibration` / `--init-zero-calibration`）。

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
| mimic | [100STYLE](https://www.ianxmason.com/100style/) | [ACM](https://dl.acm.org/doi/10.1145/3522618) | [官网](https://www.ianxmason.com/100style/) |
| mimic | Motion-X | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf) | [GitHub](https://github.com/IDEA-Research/Motion-X) |
| mimic | PHUMA | [arXiv](https://arxiv.org/abs/2510.26236) | [GitHub](https://github.com/DAVIAN-Robotics/PHUMA) |
| mimic | SOMA | [arXiv](https://arxiv.org/abs/2603.16858) | [Hugging Face](https://huggingface.co/datasets/bones-studio/seed) |
| intermimic | OMOMO | [arXiv](https://arxiv.org/abs/2309.16237) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_omomo) |
| intermimic | OmniContact-Dataset | [arXiv](https://arxiv.org/abs/2606.26201) | [Hugging Face](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) |
| meshmimic | holosoma | [arXiv](https://arxiv.org/abs/2509.26633) | [GitHub](https://github.com/amazon-far/holosoma) |
| meshmimic | PARC MS | [arXiv](https://arxiv.org/abs/2505.04002) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_parc_ms) |
| R2R | [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) | [官网](https://chingmudata.github.io/MotionDecode/) | [Hugging Face](https://huggingface.co/datasets/CMRobot/MotionDecode) |

**100STYLE** 为 Xsens MVN 风格 BVH（60 fps 风格化步态）。将解压后的目录放到名为 `100STYLE`、`xsens` 或 `xsens_mocap` 的文件夹下（例如 `assets/motions/mimic/100STYLE/`），Web 动作库即可扫描到。机器人请用参考 `xsens_mocap` 标定一次——rest 是该格式的 T-pose，不是某条 clip 的第 0 帧。单独拖入 `.bvh` 时会按关节名自动识别。

**OmniContact-Dataset** 为光学动捕人–物交互（通常 90 Hz）。请用官方 [`raw_mocap/`](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)（`motion_actor.bvh` + 物体位姿 CSV），不要用已经 retarget 到 G1 的 `npz/`。把 Hugging Face 根目录（或只把 `raw_mocap/`）放到名为 `OmniContact-Dataset` 的文件夹下，例如 `assets/motions/intermimic/OmniContact-Dataset/`。若旁边有 `assets/` 物体网格会自动匹配。重映射走 interaction-mesh（`hhtools retarget interaction-mesh` / `scripts/batch_intermimic_retarget.py --dataset omnicontact`）。默认标定参考为检测到的 BVH 方言（无法识别时用 `lafan_bvh`）。

**MotionDecode**（[ChingMu](https://huggingface.co/datasets/CMRobot/MotionDecode)）在 `samples/` 下提供已重映射到 **Unitree G1** 的 CSV（120 Hz；`root_pos_{xyz}(m)` + `root_rot` **wxyz** + `dof_*(rad)`）。这是 **机器人→机器人** 源轨迹，不是人体 mocap 适配器：请用 Web 的 **Robot → Robot**（源机器人选 `g1`），或 `scripts/batch_r2r_retarget.py`。文件没有 `time` / `# sample_rate`，必须把 **源轨迹 FPS 设为 120**（Web「源轨迹 FPS」，或 `--source-fps 120`）；默认 50 Hz 会按错误速度播放和重映射。分类子目录会按 R2R mimic clip 扫描。使用数据时请注明来源为 ChingMu。

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
- 没有这些权重时，Motion Library 的参数动作仍会以 NumPy 骨架代理正常加载和播放；真实身体网格与精确关节回归仍需要本地授权权重。
- **更多文档：** [framework.md](framework.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
