<div align="center">

<img src="hhtools/web/frontend/public/hhtools-robot.svg" alt="HHTools logo" width="88">

# Human-Humanoid Tools

**让人形机器人在约 30 秒内完成跑酷、跳舞与交互动作的重映射。**

[![Latest release](https://img.shields.io/github/v/release/Roboparty/human-humanoid-tools?include_prereleases&label=release&color=3f78c7)](https://github.com/Roboparty/human-humanoid-tools/releases/latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3f78c7.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-3f78c7.svg)](LICENSE)

[**下载**](#下载) · [**源码运行**](#源码运行) · [**文档**](#文档) · [**项目主页**](https://roboparty.github.io/human-humanoid-tools/) · [**English**](README.md)

</div>

<p align="center">
  <img src="assets/readme/demo-01.gif" width="49%" alt="重映射演示 1">
  <img src="assets/readme/demo-02.gif" width="49%" alt="重映射演示 2">
  <img src="assets/readme/demo-03.gif" width="49%" alt="重映射演示 3">
  <img src="assets/readme/demo-04.gif" width="49%" alt="重映射演示 4">
</p>

## 能做什么

<!-- &#8288; (word joiner) keeps the CJK labels on one line inside table cells. -->

| 工作流 | 流程 |
|--------|------|
| **人&#8288;体&nbsp;→&nbsp;机&#8288;器&#8288;人** | 加载动作 → 选择机器人 → 首次标定 → 重映射 → 导出。平坦动作用 Newton IK，地形与物体交互用 MPC-SQP 交互网格。 |
| **机&#8288;器&#8288;人&nbsp;→&nbsp;机&#8288;器&#8288;人** | 把已有的机器人轨迹（CSV、PKL、IsaacLab NPZ）重映射到新的 URDF。 |
| **视&#8288;频&nbsp;→&nbsp;动&#8288;作** | 用官方 [GVHMR](docs/gvhmr_cn.md) 从视频生成动作，并登记到动作库（可选，需要 CUDA）。 |
| **数&#8288;据&#8288;集&#8288;分&#8288;析** | 选择文件夹 → 扫描、打标、嵌入与聚类 → 导出子集。 |

- **任意机器人**：拖入 URDF 和 mesh，自动识别关节。
- **多种数据**：BVH、GLB 与 SMPL 系，并适配 AMASS、LAFAN、100STYLE、OMOMO、OmniContact、PHUMA [等数据集](docs/datasets_cn.md)。
- **结果可信**：诊断以 IK 实际跟踪的目标计算误差，并按机器人身高评级。
- **可自动化**：用 [CLI](docs/cli_cn.md) 批量处理，或让 Agent 通过 [JSON CLI 或 MCP](docs/agent.md) 调度任务。

## 下载

桌面版是最快的上手方式，内置 30 个动作和 7 款机器人：ROBOTO 01、Unitree G1、ROBOTO_ORIGIN、AgiBot X2、Asimov 1、Fourier GR-2 和 Berkeley Humanoid Lite。

| 平台 | 下载（0.1.6） | 须知 |
|------|---------------|------|
| **Windows** x64 | [EXE](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe) | 安装即用，无需 Python。使用自签名证书，SmartScreen 可能提示。 |
| **Linux** amd64 | [DEB](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb) | 首次启动会安装独立的 Python 环境（需要联网）。 |
| **macOS** 12+，Apple Silicon | [DMG](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg) | 首次启动同样需要安装。未经公证，需在“隐私与安全性”中允许。使用 CPU 重映射，不含视频转动作。 |

分步安装说明和校验值见[安装指南](docs/install_cn.md)，或浏览[全部版本](https://github.com/Roboparty/human-humanoid-tools/releases)。

## 源码运行

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后：

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
uv sync --locked --extra all
uv run python scripts/install_builtin_robots.py   # 只装一款可加 --only rp1
uv run hhtools web                                # 然后打开 http://127.0.0.1:8009
```

Linux 和 Windows 在有 NVIDIA GPU（CUDA 12）时会使用 GPU 加速。更小的安装方式见[安装指南](docs/install_cn.md#源码安装)。

### 运行方式

四种方式共用同一套动作、机器人与重映射核心。

| 方式 | 适用场景 | 启动 |
|------|----------|------|
| **桌面版** | 上手最快，无需命令行 | 应用菜单或 `hhtools-desktop` |
| **WebUI** | 在工作站或服务器上交互操作 | `uv run hhtools web` |
| **CLI** | 大规模数据集、SSH 与脚本 | `uv run hhtools ...` |
| **Agent** | 脚本化的 H2R、无场景 R2R 与批处理（WebUI 功能的子集） | [`hhtools agent` / `hhtools-mcp`](docs/agent.md) |

## 文档

| 指南 | 内容 |
|------|------|
| [安装](docs/install_cn.md) | 各平台桌面版安装、校验值、源码安装选项 |
| [CLI 与批处理](docs/cli_cn.md) | 命令、数据转换与导入、离线批量重映射 |
| [机器人调参](docs/robot-tuning_cn.md) | 添加机器人、`robot.yaml` 字段、常见姿态问题 |
| [数据集](docs/datasets_cn.md) | 支持的数据集、下载地址、各数据集说明 |
| [视频 → 动作](docs/gvhmr_cn.md) | 在 Linux 和 Windows 上配置 GVHMR |
| [Agent 与 MCP](docs/agent.md)（英文） | 安装、能力范围与 smoke-first 流程 |
| [任务调度](docs/job-scheduling_cn.md) | 在共享或显存紧张的 GPU 上限制并发任务 |
| [桌面打包](desktop/README.md)（英文） | 构建桌面安装包 |
| [架构](framework.md) · [贡献指南](CONTRIBUTING.md) | 代码组织方式与参与开发 |

## 反馈

欢迎提出任何建议和想法，请通过 [Issue](https://github.com/Roboparty/human-humanoid-tools/issues) 反馈。新功能需求会在核心功能稳定之后再考虑。

## 引用

若在论文或项目中使用 HHTools，请引用本仓库，并同时引用所用的上游数据集与算法（见[数据集](docs/datasets_cn.md)和 [NOTICE](NOTICE)）。

```bibtex
@software{human_humanoid_tools2026,
  title        = {human-humanoid-tools (hhtools): humanoid motion retargeting and dataset analysis},
  author       = {jaggerShen and hhtools contributors},
  year         = {2026},
  url          = {https://github.com/Roboparty/human-humanoid-tools},
  license      = {Apache-2.0}
}
```

## 许可证

代码采用 [Apache-2.0](LICENSE) 许可，第三方声明见 [NOTICE](NOTICE)。数据集不随本仓库分发。SMPL、SMPL-H、SMPL-X 权重不包含在内：请在 MPI 注册后按[人体模型说明](configs/body_models/README.md)放置。缺少这些权重时，参数动作仍会以骨架形式加载和播放，但无法得到精确的身体网格和关节回归。
