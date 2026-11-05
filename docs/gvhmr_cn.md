# 视频 → 动作（GVHMR）

**[English](gvhmr.md)**

**视频 → 动作** 使用 [GVHMR](https://github.com/zju3dv/GVHMR) 官方发布的权重完成推理，并把生成的 `hmr4d_results.pt` 登记到动作库。预处理缓存保存在动作库之外，上传的视频会统一为 GVHMR 所需的 30 FPS 时间轴。不提供自定义权重或训练入口。

请按照上游说明单独安装 GVHMR。HHTools 默认的 DEB 与 EXE 不捆绑其源码、官方 checkpoint、Python 环境或授权人体模型。明确授权的本机构建可以带入 `SMPLX_NEUTRAL.npz`；源码与 WebUI 安装仍使用本机模型目录。

macOS 桌面版不提供该工作流。

## Linux

在桌面版 **视频 → 动作** 页面点击 **配置**，选择官方 checkout、该安装对应的 Python 可执行文件，以及（单独存放时）包含 `smplx/` 的目录。这些路径保存在桌面版用户数据中，只传给独立的 GVHMR 子进程。

从终端启动时，可通过环境变量提供同样的配置：

```bash
export HHTOOLS_GVHMR_ROOT=/path/to/GVHMR
export HHTOOLS_GVHMR_PYTHON=/path/to/gvhmr/environment/bin/python
export HHTOOLS_GVHMR_BODY_MODELS=/path/to/body_models
uv run hhtools web
```

要求：

- checkout 需保留上游的 `inputs/checkpoints` 布局，其中包括已获授权的 `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`。
- 所选环境必须具备 GVHMR 依赖与 CUDA。
- `ffmpeg` 必须位于 `PATH`。

程序会尝试发现 `~/GVHMR`、仓库内的 `.venv` / `venv`，以及常见位置中名为 `gvhmr` 的 Conda 环境，但显式路径最可靠。可用 `HHTOOLS_GVHMR_TIMEOUT_SECONDS` 调整默认两小时的推理超时。

## Windows

GVHMR 是可选的 Docker 组件。用 `HHTOOLS_GVHMR_ROOT` 指向官方 checkout，用 `HHTOOLS_GVHMR_IMAGE` 指向已准备好的镜像；授权模型单独存放时，可用 `HHTOOLS_GVHMR_BODY_MODELS` 挂载。HHTools 不会安装或下载这些资源。

## 使用已有的 GVHMR 结果

已有的 GVHMR 结果可以拖入 **动作** 页面导入。转换需要本机已授权的 SMPL 系人体模型；如果它不在 HHTools 的默认搜索路径中，请用 `HHTOOLS_BODY_MODELS` 指向该目录。

纯命令行用户也可以直接把 GVHMR 输出目录转换为统一 Motion 格式：

```bash
hhtools import run --dataset gvhmr --root /path/to/gvhmr/output --out /path/to/motions
```
