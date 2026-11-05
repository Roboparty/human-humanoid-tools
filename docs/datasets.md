# Datasets

**[中文](./datasets_cn.md)**

HHTools provides adapters, not data: `assets/motions` contains demo paths only, and **no dataset is redistributed**. Download full datasets from upstream, and cite them when you publish results.

| Mode | Dataset | Paper | Download |
|------|---------|-------|----------|
| mimic | AMASS | [arXiv](https://arxiv.org/abs/1904.03278) | [site](https://amass.is.tue.mpg.de/) |
| mimic | GVHMR | [arXiv](https://arxiv.org/abs/2409.06662) | [GitHub](https://github.com/zju3dv/GVHMR) |
| mimic | LAFAN1 | [arXiv](https://arxiv.org/abs/2102.04942) | [GitHub](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) |
| mimic | [100STYLE](https://www.ianxmason.com/100style/) | [ACM](https://dl.acm.org/doi/10.1145/3522618) | [site](https://www.ianxmason.com/100style/) |
| mimic | Motion-X | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f8e27f6036c1d8b4a66b5b3a947dd7b-Paper-Datasets_and_Benchmarks.pdf) | [GitHub](https://github.com/IDEA-Research/Motion-X) |
| mimic | PHUMA | [arXiv](https://arxiv.org/abs/2510.26236) | [GitHub](https://github.com/DAVIAN-Robotics/PHUMA) |
| mimic | SOMA | [arXiv](https://arxiv.org/abs/2603.16858) | [Hugging Face](https://huggingface.co/datasets/bones-studio/seed) |
| intermimic | OMOMO | [arXiv](https://arxiv.org/abs/2309.16237) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_omomo) |
| intermimic | OmniContact-Dataset | [arXiv](https://arxiv.org/abs/2606.26201) | [Hugging Face](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) |
| meshmimic | holosoma | [arXiv](https://arxiv.org/abs/2509.26633) | [GitHub](https://github.com/amazon-far/holosoma) |
| meshmimic | PARC MS | [arXiv](https://arxiv.org/abs/2505.04002) | [Hugging Face](https://huggingface.co/datasets/YaojieShen/hhtools_parc_ms) |
| R2R | [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) | [site](https://chingmudata.github.io/MotionDecode/) | [Hugging Face](https://huggingface.co/datasets/CMRobot/MotionDecode) |

The modes map to retargeting backends: **mimic** is flat motion (Newton IK), **intermimic** is human–object interaction, **meshmimic** is terrain, and **R2R** starts from existing robot trajectories. See [CLI and batch jobs](cli.md) for commands.

## Dataset notes

### 100STYLE

Xsens MVN BVH (60 fps stylized locomotion).

- Drop the unzipped tree under a folder named `100STYLE`, `xsens`, or `xsens_mocap` (for example `assets/motions/mimic/100STYLE/`) so the Web library picks it up.
- Calibrate the robot once with reference `xsens_mocap`. Rest is the format T-pose, not a clip's first frame.
- Single-file drops are auto-detected from joint names.

### OmniContact-Dataset

Optical-mocap human–object interaction (typically 90 Hz).

- Use the official [`raw_mocap/`](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset) tree (`motion_actor.bvh` plus object-pose CSV), not the already-retargeted G1 `npz/` files.
- Place the Hugging Face root (or just `raw_mocap/`) under a folder named `OmniContact-Dataset`, for example `assets/motions/intermimic/OmniContact-Dataset/`. Object meshes are picked up from a sibling `assets/` directory when present.
- Retarget with the interaction-mesh backend (`hhtools retarget interaction-mesh` or `scripts/batch_intermimic_retarget.py --dataset omnicontact`). The default calibration reference is the detected BVH dialect (`lafan_bvh` if unknown).

### MotionDecode

[ChingMu's dataset](https://huggingface.co/datasets/CMRobot/MotionDecode) ships **Unitree G1** retargeted CSVs under `samples/` (120 Hz; `root_pos_{xyz}(m)`, `root_rot` in **wxyz**, and `dof_*(rad)`). Please credit ChingMu when you use the data.

- This is a **robot → robot** source, not a human-mocap adapter. Use the Web **Robot → Robot** panel (source robot `g1_29dof`) or `scripts/batch_r2r_retarget.py`.
- The files have no `time` or `# sample_rate`, so set **source FPS to 120** (the Web source-trajectory FPS field, or `--source-fps 120`). The default 50 Hz plays and retargets at the wrong speed.
- Nested taxonomy folders are scanned as R2R mimic clips.
