# 安装

**[English](install.md)**

- [桌面版](#桌面版)：Windows、Linux 与 macOS 安装包
- [源码安装](#源码安装)：用 uv 安装完整或精简环境

## 桌面版

从 [最新版本](https://github.com/Roboparty/human-humanoid-tools/releases/latest) 下载安装包。0.1.6 的每个安装包都内置 30 个动作和 7 款机器人。**ROBOTO 01**（`rp1`）排在 Unitree G1 前面的首位；**ROBOTO_ORIGIN**（`roboto_origin`）仍是独立模型。

三端都不默认打包 GVHMR 和受单独许可约束的 SMPL 系权重。构建说明见 [`desktop/README.md`](../desktop/README.md#desktop-packages)。

| 平台 | 安装包 | 校验文件 | 大小 |
|------|--------|----------|------|
| Windows x64 | [hhtools-0.1.6-x64-setup.exe](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-x64-setup.exe.sha256) | 433 MiB |
| Linux amd64 | [hhtools-0.1.6-linux-amd64.deb](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-linux-amd64.deb.sha256) | 233 MiB |
| macOS 12+，Apple Silicon | [hhtools-0.1.6-mac-arm64.dmg](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg) | [SHA-256](https://github.com/Roboparty/human-humanoid-tools/releases/download/0.1.6%28beta%29/hhtools-0.1.6-mac-arm64.dmg.sha256) | 287 MiB |

### Windows

1. 下载并运行 EXE。默认的当前用户安装无需管理员权限。
2. 如果 Windows 显示 SmartScreen 或未知发布者提示，属于正常情况：本预览版使用**自签名证书**，并非公众信任的商业代码签名。

安装包内置 Python、应用依赖及 Microsoft C++ 运行库，核心功能不需要源码、系统 Python，也不需要在首次启动时下载依赖。发布前已在 Windows x64 上完成安装与启动检查。

可在 PowerShell 中核对下载文件：

```powershell
Get-FileHash .\hhtools-0.1.6-x64-setup.exe -Algorithm SHA256
```

预期 SHA-256：`50b57519bfe0cce1df110385198fb9f7a795cada808794d220e65a739212f8c9`

包内 Torch 为 CPU 版本。可选的 GVHMR 视频转动作使用 Docker，需另备兼容的 NVIDIA GPU、镜像与权重，以及获授权的 SMPL-X 模型。这些资源不随包分发，本安装包也未验证 GPU 推理。

### Linux

```bash
sha256sum -c hhtools-0.1.6-linux-amd64.deb.sha256
sudo apt install ./hhtools-0.1.6-linux-amd64.deb
hhtools-desktop
```

Debian 包包含 Electron、精选内置动作与机器人。首次启动页会用同一 deb 内置的 HHTools wheel、锁定依赖清单、uv 配置和 uv 二进制安装独立运行环境。推荐的当前用户安装不需要管理员密码；所有用户安装则会打开操作系统认证窗口。

发布前已在 Linux 上完成安装与启动检查。

预期 SHA-256：`6c0a0864a197c34cc1ea05582dfeb1d14a284d8d620e4e6e6742337a669ecd8a`

### macOS

1. 打开 DMG，将 **Human-Humanoid Tools** 拖入 **应用程序**。
2. 从“应用程序”启动。此预览版使用 ad-hoc 签名，尚未经过 Apple 公证。若被 macOS 阻止，请在 **系统设置 → 隐私与安全性 → 仍要打开** 中允许。
3. 按首次启动页提示完成安装，下载独立的 Python 3.12 环境期间请保持联网。无需预装 Python、源码仓库或 Homebrew；运行环境只安装到当前用户目录，不需要管理员密码。

预期 SHA-256：`bd9df0930b1ae6d2a312ee2a0e6ccaaf66a6d4c8f16e1644cf0790beacdf5fc6`

平台限制：

- Warp 重映射使用 CPU；不支持 Intel Mac 与本机 NVIDIA GVHMR 流程。
- 桌面版隐藏视频转动作、视频导入、V2M 批处理及 GVHMR 配置入口。H2R、R2R、已有动作导入及 SMPL 系动作支持继续保留；Windows、Linux 和普通 WebUI 仍有视频转动作入口。

## 源码安装

先按照 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，再克隆仓库。HHTools 支持 Python 3.12 或更高版本；uv 会选择兼容的解释器，本机缺少时也可以安装一个。

```bash
git clone https://github.com/Roboparty/human-humanoid-tools.git
cd human-humanoid-tools
# 仅在本机没有兼容 Python 时需要：
uv python install 3.12
```

**环境要求：** Linux 与 Windows 支持 **NVIDIA GPU（CUDA 12）** 加速。视频转动作还需要单独安装、可使用 CUDA 的 [GVHMR](gvhmr_cn.md)。

### 完整环境

项目维护的 `all` extra 包含格式、机器人、重映射、WebUI 与 Agent/MCP：

```bash
uv sync --locked --extra all
uv run hhtools --help
uv run hhtools web
```

浏览器打开 `http://127.0.0.1:8009`。GVHMR 与受单独许可约束的 SMPL 系模型权重仍需自行准备；`all` 安装的是对应的 Python 集成库，不包含这些模型文件。

### 内置机器人

源码仓库只包含预设和下载配置，不包含机器人模型大文件。把需要的机器人安装到用户 Robot Library：

```bash
# 安装全部 7 款精选机器人。
uv run python scripts/install_builtin_robots.py
# 也可以只安装一款；重复 --only 可选择多款。
uv run python scripts/install_builtin_robots.py --only rp1
```

安装器下载固定提交的官方归档，只保留所选 URDF、其引用的 mesh、上游实际提供的许可证文件，以及带校验和的 `SOURCE.json`。ROBOTO 01 的 URDF 和 mesh 来自公开仓库 [`Roboparty/rp1_description`](https://github.com/Roboparty/rp1_description/tree/84529a55649b5c9796d49093c266208c879f9f29) 的固定提交；该提交（`84529a55649b5c9796d49093c266208c879f9f29`）未提供 `LICENSE` 文件。

`--replace` 会原子替换所选预设的本地修改。桌面包可以用 `HHTOOLS_BUNDLED_ROBOT_DIR` 指向审核后的 Robot Library。

### 精简环境

只安装实际需要的工作流。核心终端命令不需要 extra：

```bash
uv sync --locked
uv run hhtools --help
```

需要 WebUI 预览和 Newton 重映射时：

```bash
uv sync --locked --extra web --extra retarget
uv run hhtools web
```

只需要预览时可省略 `--extra retarget`。缺少必需的包时，启动程序会列出缺失的包和准确的修复命令，而不是直接显示 import traceback。

### 常见问题

`git pull` 之后请先 `uv sync`，再重启 `uv run hhtools web`（不要用系统里的旧包），并强制刷新浏览器。
