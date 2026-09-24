# hhtools GUI / WebUI 实测 Bug 报告

> **测试日期**:2026-08-31(续 AGENT_API_TEST_REPORT.md)
> **测试人**:AI agent(Playwright 真实驱动浏览器)
> **测试范围**:WebUI(three.js,127.0.0.1:8009)、Viser 旧版 GUI(`hhtools ui`,127.0.0.1:8008)
> **环境**:Windows 11,RTX 5060(但 venv 为 CPU-only torch),hhtools 0.1.0.dev0

## 一句话结论

**WebUI(three.js)质量非常高**——全流程实测零 JS 报错,功能链路完整;Viser 旧版 GUI 曾发现 1 个影响信息展示的真 bug,**现已修复**;另有 1 个跨端的误导性状态文案。

---

## Bug 清单

### ~~🔴 Bug G1(P1)Viser GUI Markdown 渲染失败(React error #62)~~ 【已修复】

**原现象**:Viser 页面会出现 `Markdown Failed to Render`,console 报 React error #62。部分内容在首次渲染时正常,但在搜索、进度、机器人加载、校准或 retarget 更新后再次失败。

**复核结论(2026-08-31)**:

- Viser 1.0.26 的 MDX 管道会把字符串形式的 HTML `style` 传给 React,触发 error #62。
- 原建议只包裹 `add_markdown()` 不完整:应用还有大量后续 `.content` 动态更新,同样需要经过兼容边界。
- Clip info 并非 Markdown 故障。原 disabled 文本输入框的对比度过低,数据实际存在但看起来不可见。

**修复**:

- 新增统一 Markdown 兼容层,同时清理首次创建和每次动态更新中的字符串 `style=`,并规范化 MDX 中的 `<br>`。
- 保留合法的 `style={{...}}`/`style={styleObject}` 表达式、普通文本和 `data-style`。
- Clip info 改为只读 Markdown 摘要,不改变加载和数据计算路径。
- 增加动态路径单元测试与 AST 静态守卫,防止直接 `add_markdown()` 或 `.content = ...` 回归。

**回归验证**:加载 `Xsens_mocap · stand`,切换到 G1 29dof 并加载机器人、打开已有标定确认框后,Clip info 与状态内容均可见,且没有新增浏览器错误。完整记录见 `G1_VISER_MARKDOWN_FIX.md`。

---

### ~~🟠 Bug G2(P2)"GPU×N" 状态文案在纯 CPU 环境谎称 GPU~~ 【误报,已撤销】

**复核结论(2026-08-31):不是 bug,原判断有误。**

原推断链条"torch.cuda.is_available() == False → 求解跑在 CPU"不成立:Newton IK 求解器跑在 **NVIDIA Warp** 上,而 Warp 有独立的 CUDA 栈,不依赖 PyTorch。实测:

```
>>> import warp as wp; wp.init(); wp.get_device()
warp device: cuda:0          # NVIDIA GeForce RTX 5060 (8 GiB, sm_120)
is_cuda: True
devices: ['cpu', 'cuda:0']
```

即本机 IK 确实在 GPU 上执行,"GPU×2"、"GPU-parallel Newton" 文案均**如实**。代码中的 `_warp_device_is_cuda()`(pipeline.py:198)也已做了真实设备判定。**无需修复。**原报告保留此条作为复核记录。

---

### 🔵 Bug G3(P3)上传失败无主动提示,只藏在折叠的任务历史里

**现象**:上传垃圾文件 `fake.bvh`,服务端正确拒绝(job error:`could not load fake.bvh`),但主界面**无任何 toast/弹窗/角标变化**;只有手动展开底部 Task History 才能看到失败记录(带 Retry / Duplicate & Edit)。

**影响**:用户上传坏文件后以为"没反应",可能反复重试。建议失败时给一个 toast 或让任务面板自动展开/角标变红。

---

### 🔵 Bug G4(P3)THREE.Clock 弃用警告

WebUI console 唯一一条消息:

```
[WARNING] THREE.Clock: This module has been deprecated. Please use THREE.Timer instead.
```

无功能影响,升级 three.js 后顺手改掉即可。

---

## 实测通过的功能(WebUI)

| 流程 | 结果 |
|---|---|
| 动作库浏览/搜索/加载(GLB、BVH、NPZ) | ✅ cranberry(GLB)、walk/stand(BVH)、AMASS(NPZ)均加载播放正常 |
| 3D 舞台渲染 | ✅ 截图采样 2115 种颜色,骨架/网格正常 |
| 机器人加载(6 台内置) | ✅ G1 29dof 加载,关节滑块面板正常 |
| 标定自动匹配 | ✅ Xsens walk + G1 → 自动加载 `retarget_calibration_xsens_mocap.yaml`,无标定横幅 |
| 标定模式触发 | ✅ 无标定的 tiny+G1 → 正确进入标定模式并提示对齐 |
| **H2R 全流程** | ✅ 2502 帧 IK 求解 → 完成 2487 帧 @239fps → 评估(平均 13.2cm / P95 46.9cm / 接触一致率 98% / 滑移 6.9cm/s)→ 下载 878KB CSV(2493 行,格式正确) |
| **Batch 全流程** | ✅ 库选 2 clip → G1 兼容性检查 → 并行求解 → 2 成功 → ZIP(stand.pkl 397KB + walk.pkl 359KB)自动下载 |
| **Data Analysis** | ✅ 27 clip 全库分析:质量带(ok 18 / warn 7 / bad 2)、动态带、12 种标签、20+ 指标直方图、刷选联动 |
| 任务历史持久化 | ✅ 失败/成功记录带时间戳、Retry、Duplicate & Edit |
| 错误路径:垃圾文件 | ✅ 干净拒绝 `could not load fake.bvh` |
| 错误路径:退化但合法的 BVH(1 关节 2 帧) | ✅ 正常加载不崩 |
| 错误路径:标定保存缺 motion_token | ✅ 明确报错"requires a loaded Motion — pass the clip whose frame-0 skeleton matches calibration" |
| **整个 WebUI 会话 JS 报错数** | ✅ **0**(仅 G4 一条弃用警告) |

## 实测通过的功能(Viser 旧版 GUI)

- 库索引(14 个数据集文件夹)、clip 加载(BVH)、播放/暂停/速度/时间轴均正常
- G1 修复后,clip info 可正常显示名称、帧数、FPS、Bones、坐标轴与持久化状态

## 未覆盖(需要真实素材/环境,非 bug)

- **标定保存的完整闭环**:滑块是受控组件,自动化合成事件驱动不了,需要真人拖一下再点 Save 验证;保存接口的参数校验已单独验证通过
- **Video → Motion**:需要真实视频文件 + GVHMR 环境(本机未装,"Start GVHMR" 正确保持禁用)
- **R2R 完整求解**:需要机器人轨迹源文件(面板/向导渲染正常)
