# hhtools 框架与求解原理

> **human-humanoid-tools**（包名 `hhtools`）是一套统一的人体动作导入、可视化与人形机器人重映射工具链。
> 本文档描述代码结构、数据流，以及底层 **Newton IK** 与 **MPC/SQP** 两套求解引擎的工作原理。
> 用户向的快速上手见 [README.md](README.md)。

---

## 1. 顶层目录

```
human-humanoid-tools/
├── hhtools/               # Python 包（唯一代码源头）
├── configs/               # 用户可编辑配置
│   ├── app/               # 应用默认参数
│   ├── skeleton_presets/  # 标准人形骨架与别名映射
│   └── robots/            # 机器人 preset：robot.yaml + URDF + meshes/
├── assets/                # 示例动作与导出缓存（大文件在 .gitignore）
├── framework.md           # 本文件
├── README.md              # 亮点与用法
├── LICENSE / NOTICE       # 许可证与第三方归属
└── pyproject.toml         # 包元数据与可选依赖组
```

**不入库的产物**（`.gitignore`）：`.warp_cache/`、`assets/save_npz/`、`assets/processed_npz/`、
`~/.cache/hhtools/body_models/`（SMPL 权重）、各 `__pycache__/`。

---

## 2. 包结构（自下而上）

下层模块不依赖上层；重依赖（torch、warp、mujoco、viser）一律 **lazy import**。

```
hhtools/
├── core/                  # 第 1 层：纯 NumPy 数据结构与数学
│   ├── math/              #   四元数 / 旋转 / 变换
│   ├── hierarchy.py       #   骨骼拓扑
│   ├── skeleton.py        #   静止姿态 + FK
│   ├── motion.py          #   Motion：T×J 全局位姿序列
│   ├── scene.py           #   SceneObject / TerrainHeightfield
│   ├── resample.py        #   时间重采样（位置线性 + 旋转 SLERP）
│   ├── coord.py           #   坐标系 / up-axis 转换
│   └── grounding.py       #   地面对齐与地形偏移
│
├── bodymodels/            # 第 2 层：SMPL 系列前向（可选 smplx + torch）
│   ├── params.py          #   SmplMotionParams
│   ├── engine.py          #   SmplxEngine：参数 → Motion
│   └── paths.py           #   权重发现（~/.cache/hhtools/body_models/）
│
├── io/                    # 第 3 层：格式适配
│   ├── base.py            #   load_motion / save_motion 注册表
│   ├── npz.py             #   统一 NPZ schema（v1）
│   ├── bvh.py / glb.py
│   ├── robot_csv.py       #   机器人轨迹 CSV
│   ├── parc_export.py     #   PARC MSFileData 导出
│   └── datasets/          #   公开数据集 adapter
│       ├── amass.py       #   AMASS（SMPL-H / SMPL-X）
│       ├── motion_x.py    #   Motion-X 322-dim
│       ├── omomo.py       #   OMOMO（交互物体）
│       ├── phuma.py       #   PHUMA
│       ├── hmr4d.py       #   GVHMR / 4DHumans
│       ├── meshmimic_holosoma.py  # meshmimic / holosoma 跑酷地形
│       ├── parc_ms.py     #   PARC MS 片段
│       ├── bvh_folder.py  #   LAFAN 等 BVH 目录
│       └── glb_folder.py / unified_npz_folder.py
│
├── robot/                 # 第 3 层：机器人侧
│   ├── loader.py          #   URDF → yourdfpy / MJCF
│   ├── registry.py        #   扫描 configs/robots/*
│   ├── scaffold.py        #   从 URDF 自动生成 robot.yaml
│   ├── kinematics.py      #   ik_map 拓扑推断与解剖学校验
│   └── dof_schema.py      #   CSV 列头生成
│
├── retarget/              # 第 4 层：人 → 机器人重映射
│   ├── calibration/       #   一次性标定（yaml 持久化）
│   ├── newton_basic/      #   Backend A — Newton IK（mimic）
│   └── interaction_mesh/  #   Backend B — Laplacian + MPC/SQP（intermimic / meshmimic）
│
├── viewer/                # 第 5 层：共享 UI 工具（library / anatomy / cache）
│   ├── library.py         #   assets/motions 库扫描
│   ├── anatomy.py         #   骨架分析、地面对齐、标定辅助
│   └── cache.py           #   临时 NPZ 缓存
│
├── web/                   # 第 5 层：HTML / three.js Web UI（推荐入口）
│   ├── server.py          #   FastAPI 后端
│   ├── serialize.py       #   Motion / Robot → JSON / GLB
│   └── static/            #   前端 SPA
│
└── cli/                   # Typer CLI（convert / import / robot / retarget / web）
```

---

## 3. 核心数据流

```
原始输入                    统一中间表示              重映射输出
─────────                  ────────────              ──────────
BVH / GLB  ──┐
SMPL 参数序列    ──┼──►  Motion + NPZ  ──►  RetargetedMotion  ──►  robot CSV / PARC pkl
公开数据集目录   ──┘         │                      ▲
                            │                      │
                            ▼                      │
                     Web UI 可视化            calibration
                     （three.js）              + scaler
                                               + IK / SQP
```

**关键不变量**：

| 字段 | 约定 |
|------|------|
| `Motion.positions` | 米，世界坐标，`up_axis="Z"`，`forward="+X"` |
| `Motion.quaternions` | 全局四元数，**xyzw** |
| 机器人 CSV | `time, root_xyz, root_qxyzw, dof_<j1>, dof_<j2>, …` |

---

## 4. 重映射模式与后端选择

| 模式 | 场景 | 后端 | 模块 |
|------|------|------|------|
| **mimic** | 纯人体骨架（跳舞、行走、LAFAN 等） | Newton IK | `retarget/newton_basic/` |
| **intermimic** | 人体 + 刚性交互物体（OMOMO 等） | Laplacian + MPC/SQP | `retarget/interaction_mesh/` |
| **meshmimic** | 人体 + 地形高度场（跑酷障碍） | 同上 + 地形碰撞 | `retarget/interaction_mesh/` |

Web UI 根据动作是否携带 `objects` / `terrain` 自动推荐后端；用户也可手动切换。

---

## 5. Newton IK 引擎（`newton_basic`）

> 改编自 NVIDIA [SOMA-Retargeter](https://github.com/NVlabs/SOMA-Retargeter)（Apache-2.0）。

### 5.1 管线概览

每帧按序执行：

```
Motion
  │  ① HumanToRobotScaler（按标定 yaml 缩放效应器目标）
  ▼
(F, M, 7) effector targets   # M = ik_map 映射关节数，7 = pos + quat
  │  ② FeetStabilizer（落脚点锁定、接地约束）
  ▼
constrained targets
  │  ③ Newton IK（NVIDIA Warp GPU 加速，可选 CUDA Graph）
  ▼
(F, nq) joint_q
  │  ④ JointLimitClamper + 速度限幅
  ▼
RetargetedMotion
```

### 5.2 Scaler（人体 → 机器人尺度）

`HumanToRobotScaler` 是纯 NumPy 的 Stage-1 模块，将源骨架全局位姿映射为机器人效应器目标：

- 按 `ScalerConfig.joint_scales` 对每个映射关节做**位置缩放 + 旋转保持**；
- 标定 yaml 提供 `human_height`、`robot_height`、逐关节偏移与缩放因子；
- LAFAN 等数据集可选 toe-orientation 模式（脚踝位置 + 脚趾朝向）。

### 5.3 Newton IK 求解

基于 [Newton Physics](https://github.com/newton-physics/newton) + [NVIDIA Warp](https://github.com/NVIDIA/warp)：

- 为每个 `ik_map` 条目创建 `IKObjectivePosition` / `IKObjectiveRotation` 残差；
- 附加 `IKSmoothJointFilter`（软限位偏好，防止肩/腰漂移）与 `IKObjectiveJointLimit`（硬限位）；
- 默认在 CUDA 上用 `wp.ScopedCapture` 录制 IK graph，逐帧 replay 以消除 kernel 启动开销；
- CPU 或无 GPU 时回退到逐帧 `solver.step`。

**为何快**：单帧 IK 问题规模小（~20 DOF、~14 效应器），GPU 批处理 + CUDA Graph 使 30 s 级片段可在数秒内完成。

### 5.4 后处理

- `FeetStabilizer`：检测触地帧，锁定脚部位姿防止滑步；
- `JointLimitClamper`：硬裁剪到 URDF 关节限位；
- 速度限幅：相邻帧 Δq 平滑，消除 IK 抖动。

---

## 6. MPC/SQP 引擎（`interaction_mesh`）

> 拉普拉斯交互网格思路参考 [holosoma](https://github.com/NVIDIA-Omniverse/holosoma) interaction-mesh retargeting（Apache-2.0）。

### 6.1 为何需要第二套引擎

纯 IK 只跟踪骨架效应器，**无法保证**：

- 人体与交互物体之间的相对空间关系（搬箱子、推椅子）；
- 脚/手与地形的非穿透约束（跑酷越障）。

交互网格后端把人体关节、物体采样点、地形采样点拼成一张 **Delaunay 四面体网格**，用拉普拉斯坐标保持局部几何关系，再用 SQP 在 MuJoCo 关节空间中求解。

### 6.2 管线概览

```
Motion + objects/terrain
  │  ① 均匀缩放（robot_height / human_height，保持人体比例）
  ▼
ScaledMotionScene
  │  ② 预计算每帧目标拉普拉斯坐标（Delaunay 拓扑 + Laplacian target）
  ▼
List[FrameLaplacianTarget]
  │  ③ 逐帧 SQP 循环（iterate_mpc）
  ▼
  ┌─ 构造二次子问题：min ½·dq'P·dq + q'·dq
  │    目标项：拉普拉斯跟踪 + 平滑 + 源姿态跟踪
  │    约束项：关节盒约束 + 非穿透（OSQP）+ 脚/手接触
  └─ MuJoCo FK + Jacobian 线性化
  ▼
(F, nq) joint_q  →  RetargetedMotion
```

### 6.3 拉普拉斯交互网格

对每帧 \(f\)：

1. 取缩放后的人体关节位置 \(\{h_i\}\)；
2. 对交互物体/地形采样 \(\{o_j\}\)，拼成顶点集 \(V_f\)；
3. 以中间帧为 pivot 做 Delaunay 四面体剖分，得到共享邻接表 `adj_list`；
4. 计算目标拉普拉斯坐标 \(\delta^*_f = L \cdot V_f\)（umbrella 权重）。

求解时，机器人对应 body 上的点通过 MuJoCo Jacobian \(J_V\) 线性化：

\[
J_L = \mathrm{kron}(L, I_3) \cdot J_V, \quad
\min_{dq} \| J_L \cdot dq - (\delta^* - \delta_0) \|^2
\]

### 6.4 SQP / MPC 子问题

`qp_step.py` 实现无 cvxpy 的 QP 求解：

| 方法 | 场景 |
|------|------|
| 稠密 KKT / L-BFGS-B | 默认；纯二次 + 盒约束 |
| OSQP | 含不等式（非穿透、接触）时 |

外层 `iterate_mpc` 以 trust-region 迭代：

1. 在当前 \(q\) 处 MuJoCo `mj_forward` + 构建 \(J_V\)；
2. 解 QP 得 \(\Delta q\)，更新 \(q \leftarrow q + \Delta q\)；
3. OSQP 失败时缩小 trust-region（`OSQP_FALLBACK_TRUST_SHRINK = 0.25`）回退到盒约束 L-BFGS-B；
4. 重复至残差收敛或达最大迭代。

地形通过 `heightfield.py` 将 OBJ sidecar 编译为 MuJoCo hfield；`collision.py` 添加足/手-地面非穿透行。

### 6.5 与 Newton IK 的分工

| | Newton IK | MPC/SQP |
|---|-----------|---------|
| 输入 | 骨架效应器目标 | 人体 + 物体/地形顶点 |
| 约束 | 关节限位 + 脚稳定 | 拉普拉斯 + 碰撞 + 接触 |
| 速度 | 极快（GPU IK graph） | 较慢（逐帧 SQP + MuJoCo FK） |
| 适用 | mimic | intermimic / meshmimic |

两套后端共享 `calibration/` 标定与 `HumanToRobotScaler` 配置推导逻辑。

---

## 7. Web UI 架构

```
浏览器（three.js）          FastAPI（hhtools/web/server.py）
     │                              │
     │  REST / WebSocket            │
     ├──── motion/upload ──────────►│ io.load_motion → Motion
     ├──── robot/upload ───────────►│ robot.scaffold → URDFRobotModel
     ├──── calibration/save ───────►│ retarget.calibration
     ├──── retarget/run ───────────►│ newton_basic / interaction_mesh pipeline
     └──── export/csv ─────────────►│ io.robot_csv
```

前端负责 3D 渲染与交互标定；后端复用完整 `hhtools` 管线，数据不离开本机。

---

## 8. 扩展点

| 需求 | 做法 | 入口 |
|------|------|------|
| 新文件格式 | 实现 `MotionLoader`，`register_loader(".ext", fn)` | `hhtools.io.base` |
| 新数据集 | 继承 `DatasetAdapter`，`@register_dataset` | `hhtools.io.datasets.base` |
| 新机器人 | URDF + meshes → `configs/robots/<name>/`，`hhtools robot validate` | `hhtools.robot.registry` |
| 新重映射后端 | 实现 `pipeline.run(motion) → RetargetedMotion` | `hhtools.retarget` |
| 新 Web 端点 | `web/server.py` 加路由 + `serialize.py` | `hhtools.web` |

---

## 9. 可选依赖组

| extra | 作用 |
|-------|------|
| `web` | FastAPI + three.js UI + SMPL + MuJoCo + OSQP |
| `retarget` | Newton IK（warp-lang + newton） |
| `retarget-interaction` | 交互网格 MPC/SQP |
| `formats` / `smpl` / `robot` | 按需拆分 |
| `all` | 一次安装全部应用依赖 |

---

## 10. 许可证与上游归属

### 10.1 本仓库

- 代码：[Apache-2.0](LICENSE)
- 第三方组件清单：[NOTICE](NOTICE)

### 10.2 算法与代码参考

| 上游 | 许可 | 本仓库使用 |
|------|------|-----------|
| [SOMA-Retargeter](https://github.com/NVlabs/SOMA-Retargeter) | Apache-2.0 | Newton IK 管线、Scaler、FeetStabilizer |
| [holosoma](https://github.com/NVIDIA-Omniverse/holosoma) | Apache-2.0 | 拉普拉斯交互网格、MPC/SQP 公式 |
| [ai4animationpy](https://github.com/facebookresearch/ai4animationpy) | Apache-2.0 | NPZ schema 与 API 形态参考（代码全新编写） |
| [Newton](https://github.com/newton-physics/newton) | Apache-2.0 | IK 求解器 |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | Apache-2.0 | GPU 加速 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | 交互场景 FK / 碰撞 |
| [smplx](https://github.com/vchoutas/smplx) | Apache-2.0 | SMPL 前向 |

### 10.3 人体模型权重（须用户自行下载）

| 模型 | 发布方 | 许可 |
|------|--------|------|
| SMPL | Max Planck Institute | 非商业科研 |
| SMPL+H / MANO | MPI | 非商业科研 |
| SMPL-X | MPI | 非商业科研 |

权重**不随本仓库分发**。下载后放至 `~/.cache/hhtools/body_models/`。

### 10.4 公开数据集（须用户自行获取）

| 数据集 | 上游 | 备注 |
|--------|------|------|
| AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de) | SMPL-H/X 动捕，非商业 |
| Motion-X | [IDEA-Research/Motion-X](https://github.com/IDEA-Research/Motion-X) | SMPL-X 322-dim |
| OMOMO | [lijiaman/omomo_release](https://github.com/lijiaman/omomo_release) | 人体+物体交互 |
| PHUMA | [DAVIAN-Robotics/PHUMA](https://github.com/DAVIAN-Robotics/PHUMA) | 精选人体动作 |
| GVHMR | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | 视频单目 HMR |
| meshmimic / holosoma | [NVIDIA-Omniverse/holosoma](https://github.com/NVIDIA-Omniverse/holosoma) | 跑酷地形片段 |
| LAFAN1 | [ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | BVH 动捕 |
| PARC MS | holosoma 生态 | 地形+动作 pickle |

本工具仅提供**格式适配器**，不重新分发数据集文件。使用前请阅读各上游 License 并遵守其条款。

---

## 11. 已清理的冗余模块

以下目录/文件已从代码树移除（空占位、一次性脚本或测试专用）：

- `tests/` — pytest 套件（与生产代码解耦，不再入库）
- `scripts/` — 一次性批处理 / smoke 脚本
- `docs/` + `mkdocs.yml` — 文档站（内容已收敛至本文件与 README）
- 历史空包：`hhtools/analytics/`、`hhtools/backend/`、`hhtools/robot/adapters/`
- stub CLI：`hhtools/cli/analyze.py`

`hhtools/viewer/` 保留：`web/` 后端复用其 `library`、`anatomy`、`cache` 模块；Viser 旧 UI（`hhtools ui`）仍可用但不再是推荐入口。
