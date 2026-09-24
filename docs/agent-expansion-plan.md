# HHTools CLI 与 Agent 能力扩展计划

## 目标

在保持现有 H2R Agent 安全边界和兼容性的前提下，逐步让 CLI、REST 与 MCP
覆盖 HHTools 的核心工作流。首轮范围包括：

- revision-aware 作业等待；
- H2R Interaction-Mesh 正式验收；
- Robot-to-Robot（R2R）；
- H2R / R2R Batch；
- H2R 标定状态、候选、验证、视觉预览与静默保存。

Video-to-Motion 和 Analysis 暂不进入本轮实现。

## 实施状态

- [x] 阶段 1：revision-aware `wait_job`（service、REST、MCP、JSON CLI、Agent skill）
- [x] 阶段 2：H2R Interaction-Mesh 正式验收（terrain 与 object 自包含 E2E）
- [x] 阶段 3：scene-free R2R（不可变 robot pair identity 与自包含 MCP E2E）
- [x] 阶段 4：可配置 H2R / R2R Batch（默认不限，共享生命周期与双工作流 E2E）
- [x] 阶段 5：H2R 标定辅助与 GPT 视觉自动标定

## 设计原则

1. **一个作业生命周期**：不同工作流拥有各自的 preflight，但共用
   `start / wait / get / cancel / retry / artifacts`。
2. **先计划后执行**：任何求解或批处理都必须由不可变 plan 启动，并使用调用方提供的
   idempotency key。
3. **人机边界明确**：Agent 可以检查标定、生成候选并验证候选；确定性验证通过后允许
   静默保存。GPT 视觉模式额外审查前/侧视图并记录声明，但模型名称不是鉴权凭据；full run
   与真实机器人部署仍保留独立的人类确认。
4. **接口兼容**：现有 `preflight_retarget`、`start_retarget`、REST 路径和 JSON CLI
   在 v1 生命周期内保留。
5. **结果可移植**：默认响应保持紧凑，不嵌入轨迹、视频、网格或宿主机绝对路径；大内容
   继续通过受校验 artifact 暴露。
6. **共享实现**：MCP、REST、JSON CLI 和未来的人类 CLI 只做适配，不复制求解、标定、
   调度或产物逻辑。

## 当前能力基线

| 能力 | 当前状态 | 本轮目标 |
| --- | --- | --- |
| H2R / Newton | 已支持 | 保持兼容 |
| H2R / Interaction-Mesh | 底层部分接通，缺少正式 E2E 与文档 | 正式验收 |
| R2R | Web 支持，Agent 未暴露 | 增加 Agent preflight 与执行 |
| H2R / R2R Batch | Web 支持，Agent 未暴露 | 增加默认不限、可配置的批处理计划与聚合结果 |
| Video-to-Motion | 仅 Web / 独立 GVHMR 环境 | 暂缓 |
| Analysis | Web 支持 | 暂缓 |
| 自动标定 | 仅人工 GUI 标定 | 增加状态、候选、验证、视觉预览与经验证的静默保存 |

## 目标接口

### 共享作业接口

```text
start_job(plan_id, idempotency_key)
wait_job(job_id, after_revision, timeout)
get_job(job_id, after_revision?)
lookup_job(plan_id, idempotency_key, after_revision?)
cancel_job(job_id)
retry_job(job_id, idempotency_key)
list_job_artifacts(job_id, limit, offset)
export_artifact(job_id, artifact_id)
```

现有 `start_retarget` 作为兼容别名保留。新增工作流不再复制一套作业查询、取消和产物接口。

### 工作流 preflight

```text
preflight_h2r(request)
preflight_r2r(request)
preflight_batch(request)
```

现有 `preflight_retarget` 作为 `preflight_h2r` 的兼容名称保留。

### 标定辅助

```text
get_calibration_status(request)
propose_calibration(request)
validate_calibration(request)
preview_calibration(request)
save_calibration(request)
get_r2r_calibration_status(request)
propose_r2r_calibration(request)
validate_r2r_calibration(request)
preview_r2r_calibration(request)
save_r2r_calibration(request)
```

`propose_calibration` 只生成可审查候选，不写入正式 calibration。候选包含输入资产
identity、算法版本、锁定关节和验证结果。`preview_calibration` 返回可供 GPT 视觉模型直接检查
的前/侧视 PNG。`save_calibration` 仅接受当前仍通过确定性验证的候选；
`gpt_vision_silent` 还要求一份通过的视觉审查声明，并将候选与审查来源写入标定记录。
客户端自报的模型名称只作为审计提示，不作为权限或真实性判断。

## 阶段 1：`wait_job`

### 契约

```text
wait_job(
    job_id: str,
    after_revision: int,
    timeout: float = 30,
) -> AgentJobView
```

- `timeout` 必须有限且有上限，首版范围为 `0..60` 秒；
- 作业已 terminal 时立即返回；
- 当前 revision 大于 `after_revision` 时立即返回；
- revision 未变化时等待，超时后返回当前快照；调用方通过 revision 是否变化判断超时；
- `after_revision` 不能大于当前 revision；
- 取消一次等待不得取消作业；
- 作业状态变化应通过条件变量或事件唤醒，不使用高频 sleep 轮询；
- MCP、REST 与 JSON CLI 必须共享同一 service 方法和错误契约。

REST 兼容路径：

```text
GET /api/agent/v1/jobs/{job_id}/wait?after_revision=N&timeout=30
```

JSON CLI：

```text
hhtools agent job wait JOB_ID --after-revision N --wait-timeout 20
```

JSON CLI 的既有 `--timeout` 表示整个 HTTP 请求超时，因此等待时要求它大于
`--wait-timeout`；MCP 与 REST 参数仍使用 `timeout`。

### 验收

- revision 已更新、超时、terminal、未知 job、非法 revision 和非法 timeout 均有测试；
- 多个等待者能被同一次进度更新唤醒；
- 等待者取消或超时后不遗留线程和锁；
- MCP live schema、REST OpenAPI 与 JSON CLI help 都声明相同边界；
- 原有 `get_job` 行为不变。

## 阶段 2：H2R Interaction-Mesh 正式验收

- [x] 补齐 object-interaction 与 terrain-scene 的自包含 MCP Agent E2E fixtures；
- [x] 验证 backend 自动路由、manual calibration、CSV 场景 ZIP artifact；
- [x] 验证 Interaction-Mesh 的协作取消、结构化失败报告与 execution provenance；
- [x] 验证 portable export receipt、相对路径、hash 与场景 sidecar 内容；
- [x] 更新 capability 验收、Agent 文档、skill 路由说明和示例。

## 阶段 3：R2R

- [x] 定义 source trajectory、source robot、target robot 与 pair calibration 的不可变 identity；
- [x] 增加安全 R2R asset inspection、可发现目录项与 `preflight_r2r`；
- [x] 复用共享 JobManager / `start_job`，新增 R2R executor adapter；
- [x] 对 source/target 不匹配、缺少 pair calibration、过期计划和场景输入进行前置拒绝；
- [x] 产物沿用现有 CSV / PKL、preview、diagnostics 与 portable export，并补齐 provenance；
- [x] MCP、REST 和 JSON CLI 增加 R2R 契约并完成自包含真实求解 E2E；
- [ ] 人类 CLI 的 `hhtools retarget r2r` 简洁入口留在第 6 个提交完成。

## 阶段 4：Batch

- [x] 首版只覆盖 H2R 与 scene-free R2R Batch；
- [x] `preflight_batch` 冻结有序 child plan、每项 hash、机器人、标定、backend、输出策略和可选资源策略；
- [x] 明确 `success / partial / review_required / rejected` 聚合语义；
- [x] JobProgress 提供 `completed_items / total_items`，默认响应不包含无界逐项数组；
- [x] 完整逐项结果写入 `batch_report`，失败同步进入 failure report，输出汇总为 portable ZIP；
- [x] 取消停止未启动条目，并在当前 child 的安全边界协作取消；
- [x] retry 创建新的 whole-batch child attempt，不修改原批任务或隐式选择失败子集；
- [x] MCP、REST、JSON CLI 与 H2R/R2R 自包含 E2E 完成验收；
- [x] Batch 条目/总帧设置默认 `0 = 不限`；正数限制可持久化并在运行中热更新；
- [ ] 人类 CLI 的 `hhtools batch h2r` / `hhtools batch r2r` 留在第 6 个提交统一完成。

## 阶段 5：H2R / R2R 标定辅助

- [x] `get_calibration_status`：只读返回 reference、来源、hash、映射与当前质量；
- [x] `propose_calibration`：通过 URDF 拓扑、参考姿态与关节限位生成可修订候选；
- [x] `validate_calibration`：计算关节限制、关键肢段方向、足底与左右对称检查；
- [x] `preview_calibration`：以 MCP image content 返回确定性的前/侧视 PNG；
- [x] `save_calibration`：确定性验证通过后允许 `validated_silent`；GPT 视觉审查通过后允许
  `gpt_vision_silent`，并始终写入用户 overlay；覆盖前归档旧版本、拒绝变化过的 baseline，
  避免改变已登记 robot bundle 或静默丢失并发修改；
- [x] 候选以内容寻址方式持久化，可通过 parent candidate、关节覆盖和锁定关节迭代；
- [x] preflight 缺少标定时仍返回 `human_action_required`，支持自动标定的 Agent 完成后必须
  重新 preflight；
- [x] GUI 能生成候选并在现有 3D 标定编辑器中继续人工微调与保存；
- [x] 自动标定不授权 full run 或真实机器人部署。
- [x] R2R 标定绑定 source/target 两个 robot bundle，以 source 零位 FK 作为参考骨架，候选只
  修改 target joint pose；独立 MCP 工具提供 status/propose/validate/preview/save 闭环；
- [x] 缺少 pair calibration 时 `preflight_r2r` 返回精确的
  `get_r2r_calibration_status` Agent action；保存后必须重新 preflight；
- [x] R2R 静默保存固定写入 target 用户 overlay，并继承内容寻址候选、旧版本归档、并发
  baseline 拒绝、幂等重放与 GPT 视觉审查约束。

## CLI 首页目标

```text
Workflows
  Human -> Robot
  Robot -> Robot
  Batch

Tools
  Convert Motion
  Robots
  System Check

Open
  WebUI
  Desktop GUI
```

完整参数调用保持非交互；真实 TTY 中缺少参数时可以进入简洁向导。Agent 和 CI 使用
`--json` 或 MCP，永不收到交互提示、ANSI 进度或非结构化第二份输出。

- [x] `hhtools` 无参数启动时显示响应式首页，窄终端自动改为纵向布局；
- [x] 首页只链接当前真实可用入口，R2R / Batch 暂时明确导向 WebUI；
- [x] 非 TTY 输出不含 ANSI、不读取输入，`--help` 与 `--version` 行为保持不变；
- [ ] R2R / Batch 人类 CLI 与真实 TTY 简洁向导在后续步骤接入首页。

## 提交与发布顺序

1. `feat(agent): add revision-aware job waiting`
2. `test(agent): verify interaction-mesh execution`
3. `feat(agent): add robot-to-robot workflow`
4. `feat(agent): add preflighted batch workflows`
5. `feat(agent): add calibration proposals`
6. `feat(cli): expose r2r and batch workflows`

每个阶段必须独立通过 Python 回归、MCP live schema 测试、REST 测试、JSON CLI 测试、
Ruff 和 wheel 内容检查；不得通过扩大 legacy baseline 或跳过关键测试来换取绿灯。
