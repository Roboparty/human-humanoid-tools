# 前端视觉规范

## 定位

- 参考 ChatGPT/Codex 应用的克制感：平铺工作区、小圆角、细分隔线、紧凑控件、低视觉噪声。
- 颜色只采用 Roboparty Lab 的品牌色，不复制 ChatGPT 的品牌配色。
- HHTOOLS 是桌面工具，不做成营销页面，也不堆叠浮动卡片。

## 颜色

| 语义 | Token | 色值 |
|---|---|---|
| 应用底色 | `--canvas` | `#F6F8FA` |
| Stage 画布底色 | `--stage-canvas` | `#F5F5F7` |
| 主表面 | `--surface` | `#FFFFFF` |
| 浅表面 | `--surface-subtle` | `#F1F4F7` |
| 主文字 | `--text` | `#02122E` |
| 次要文字 | `--text-muted` | `#667085` |
| 浅边框 | `--border-subtle` | `#E5EAF0` |
| 强边框 | `--border` | `#D0D7DE` |
| 强调色 | `--accent` | `#0071E3` |
| 强调色 hover | `--accent-hover` | `#005BB5` |
| 强调浅底 | `--accent-soft` | `#E8F2FF` |
| 焦点环 | `--focus` | `#218BFF` |

深色文字保留 Roboparty Lab 的品牌基因；交互蓝回到旧版 HHTOOLS 的鲜明方向。
中性色与语义分层参考 Primer，焦点与工具栏对比参考 VS Code；不直接复制任何完整主题。
成功、警告和错误色等真实状态出现后再单独定义。

## 形状与间距

- 圆角只用三级：紧凑控件 `4px`，输入框和按钮 `6px`，菜单和弹窗 `8px`。
- 只有图标按钮、头像和状态点可以使用全圆角；普通文字控件不做成胶囊。
- 间距遵循 `4px` 网格：`4 / 8 / 12 / 16 / 24`。
- 工具栏与列表行高保持在 `32-36px`，顶部菜单栏保持 `40px`。
- 区域边界使用低对比度 `1px` 线；只有菜单和弹窗使用轻阴影。

## 布局与文字

- 大区域使用平面分隔，不把每个区域包成卡片，禁止卡片嵌套。
- 工具视图优先使用可用宽度；阅读型内容才限制行宽。
- 窄屏 WebUI 收拢左栏；空间不足时由 Inspector 取代空白 Stage，不允许横向溢出。
- 字体顺序：Inter、Noto Sans SC、Microsoft YaHei、系统无衬线字体。
- 导航与控件使用 `12-14px`，正文使用 `14-16px`；标题保持克制，不随视口缩放。

## 交互

- hover 使用 `--surface-subtle`；选中态可增加 `--accent` 文字或小型指示条。
- focus 必须清晰可见并使用 `--focus`，不能只依赖颜色变化。
- disabled 同时降低文字和图标对比度，并取消 hover 反馈。
- 菜单、弹窗和页面状态由 React state 控制，不查询或修改无关 DOM。
- 动画只服务于状态变化，并遵守 `prefers-reduced-motion`。

## 实施规则

- 组件只使用语义 token，不直接写品牌色 hex。
- `styles.css` 只保存 token 和页面级基础；组件私有样式写在组件内的 Tailwind class。
- shadcn 组件按需加入，并统一映射到本规范的 token。
- 有标准含义的图标统一使用 Lucide。
- 不使用渐变装饰、彩色光球、重阴影、超大标题和应用内说明文案。
- 第一阶段只定义浅色主题；深色主题必须单独设计，不能自动反色生成。
- Stage 图层菜单保留旧版蓝、青绿、紫色的对象语义，只将外框改为 `8px` 圆角。

## 进度

- [x] 确定视觉方向与颜色 token
- [x] 将现有菜单壳迁移到新 token 与小圆角
- [x] 按相同规范建立固定左侧功能导航
- [x] 使用 Tailwind + shadcn ToggleGroup 建立舞台左上悬浮菜单
- [x] 将组件私有样式迁入组件内 Tailwind class
- [x] 按旧版信息层级建立 Motion 右侧面板样式壳
- [x] 按相同视觉规则建立 Robot 右侧面板样式壳
- [x] 建立 Video → Motion pipeline 与步骤样式壳
- [x] 建立 Human → Robot pipeline 与步骤样式壳
- [x] 建立 Robot → Robot pipeline 与步骤样式壳
- [x] 建立 Batch 三模式右侧面板样式壳
- [x] 建立 Data Analysis pipeline 与步骤样式壳
- [x] 七个功能页面的视觉壳均遵循本规范
- [x] 中间 Stage 迁移为单一 R3F Canvas，保留旧相机、灯光、网格和坐标轴
- [x] R3F Canvas 使用旧版 `NoToneMapping` 色彩管线，避免材质颜色漂移
- [x] 迁移旧版蓝色 source skeleton 的首帧视觉层
