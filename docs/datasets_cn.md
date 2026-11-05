# 数据集

**[English](datasets.md)**

HHTools 只提供格式适配，不提供数据：`assets/motions` 中只有演示片段，**不重新分发任何数据集**。完整数据请从上游下载，发表成果时请引用对应数据集。

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

模式对应不同的重映射后端：**mimic** 为平坦动作（Newton IK），**intermimic** 为人–物交互，**meshmimic** 为地形，**R2R** 从已有机器人轨迹开始。命令见 [CLI 与批处理](cli_cn.md)。

## 各数据集说明

### 100STYLE

Xsens MVN 格式 BVH（60 fps 风格化步态）。

- 将解压后的目录放到名为 `100STYLE`、`xsens` 或 `xsens_mocap` 的文件夹下（例如 `assets/motions/mimic/100STYLE/`），Web 动作库即可扫描到。
- 机器人请用参考 `xsens_mocap` 标定一次。rest 是该格式的 T-pose，不是某条 clip 的第一帧。
- 单独拖入的文件会按关节名自动识别。

### OmniContact-Dataset

光学动捕人–物交互（通常 90 Hz）。

- 请使用官方 [`raw_mocap/`](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)（`motion_actor.bvh` 加物体位姿 CSV），不要用已经重映射到 G1 的 `npz/` 文件。
- 把 Hugging Face 根目录（或只把 `raw_mocap/`）放到名为 `OmniContact-Dataset` 的文件夹下，例如 `assets/motions/intermimic/OmniContact-Dataset/`。旁边有 `assets/` 目录时会自动匹配物体网格。
- 重映射使用 interaction-mesh 后端（`hhtools retarget interaction-mesh` 或 `scripts/batch_intermimic_retarget.py --dataset omnicontact`）。默认标定参考为检测到的 BVH 方言（无法识别时用 `lafan_bvh`）。

### MotionDecode

[ChingMu 的数据集](https://huggingface.co/datasets/CMRobot/MotionDecode) 在 `samples/` 下提供已重映射到 **Unitree G1** 的 CSV（120 Hz；`root_pos_{xyz}(m)`、**wxyz** 顺序的 `root_rot`，以及 `dof_*(rad)`）。使用数据时请注明来源为 ChingMu。

- 这是 **机器人 → 机器人** 的源轨迹，不是人体动捕适配器。请使用 Web 的 **机器人 → 机器人** 面板（源机器人选 `g1_29dof`），或 `scripts/batch_r2r_retarget.py`。
- 文件没有 `time` 或 `# sample_rate`，因此必须把 **源轨迹 FPS 设为 120**（Web 的“源轨迹 FPS”，或 `--source-fps 120`）。默认的 50 Hz 会以错误的速度播放和重映射。
- 分类子目录会按 R2R mimic clip 扫描。
