# G1：Viser GUI Markdown 渲染失败——修复记录

> **日期**：2026-08-31
> **来源**：`GUI_WEBUI_TEST_REPORT.md` · Bug G1（P1）
> **状态**：已修复并完成回归验证
> **范围**：仅修改 Viser GUI 的内容渲染兼容层与 Clip info 展示，不修改标定、IK、retarget 或资产加载算法

---

## 1. 问题现象

旧版 Viser GUI 启动后会出现以下现象：

- 浏览器 console 报 React error #62；
- 多处说明或状态区域显示 `Markdown Failed to Render`；
- 首次打开页面时部分区域可能正常，但搜索、加载机器人、校准或 retarget 等操作更新内容后再次报错；
- Clip info 的名称、帧数、FPS 和 Bones 数据看起来不可见。

这里实际包含两个独立问题：Markdown/MDX 兼容问题，以及禁用状态文本的对比度问题。

## 2. 根因

### 2.1 字符串形式的 `style` 与旧版 Viser 的 MDX 管道不兼容

项目锁文件当前使用 Viser 1.0.26。该版本通过 MDX `evaluate()` 渲染 Markdown。源码中存在如下 HTML：

```html
<span style='color:red'>...</span>
```

MDX 将字符串形式的 `style` 传给 React，而 React DOM 要求 `style` 是样式对象，因此触发 React error #62。

此外，部分动态内容包含 `<br>`。在 MDX/JSX 语境中它需要写成自闭合的 `<br/>`，否则机器人加载后的确认对话框会产生新的 MDX 解析错误。

### 2.2 原方案只包裹创建操作，无法覆盖动态更新

`hhtools/viewer/app.py` 原来有：

- 15 处 Markdown 创建调用；
- 42 处通过 `.content = ...` 执行的动态更新。

仅把 `add_markdown()` 替换为创建 wrapper，只能清理第一次传入的内容。后续直接赋值给 `.content` 时仍会绕过 wrapper，使问题在交互后复发。

### 2.3 Clip info 是独立的低对比度问题

Clip info 原先使用四个 `add_text(..., disabled=True)` 控件。数据加载与更新链路是正常的，但 Viser/Mantine 会降低 disabled 控件的不透明度，导致文字和背景的对比度过低，看起来像是没有值。

因此，Markdown wrapper 本身不能修复 Clip info。

## 3. 实际修复

### 3.1 建立统一的 Viser Markdown 兼容边界

新增 `hhtools/viewer/markdown_compat.py`，提供三个显式接口：

- `sanitize_markdown_for_viser()`：仅清除标签中的字符串 `style=` 属性，并把 `<br>` 规范化为 `<br/>`；
- `add_safe_markdown()`：清理首次内容后调用 Viser `add_markdown()`；
- `set_safe_markdown()`：清理动态内容后再更新真实 handle 的 `.content`。

该兼容层有意保持最小范围：

- 不改普通 Markdown 和普通文本；
- 不误删 `data-style`；
- 不误删正文中的 `style='...'` 字样；
- 保留 MDX 合法的 `style={{...}}` 与 `style={styleObject}` 表达式；
- 返回真实 Viser handle，不引入代理对象，避免改变事件、可见性与生命周期行为。

### 3.2 同时覆盖首次创建和动态更新

`hhtools/viewer/app.py` 中所有 Markdown 创建都改用 `add_safe_markdown()`，所有 `.content` 更新都改用 `set_safe_markdown()`。

源内容中的样式字符串暂时保留。这样兼容逻辑集中在边界层，将来升级到不再受此问题影响的 Viser 渲染管道时，可以单独移除兼容层，而不需要反向恢复每段 UI 文本。

### 3.3 将 Clip info 改为只读 Markdown 摘要

四个 disabled 文本输入框被替换为一个只读、无内联样式的 Markdown 摘要，继续显示相同数据：

- Name；
- Frames · FPS · Bones；
- Up axis（source → view）；
- Persisted 状态。

原有加载、缓存和数据计算路径不变，只调整最终展示控件。

## 4. 回归保护

新增 `tests/viewer/test_markdown_compat.py`，覆盖：

- 单引号、双引号、大小写和无引号的字符串 `style`；
- 同一标签多个 `style`；
- `<br>` 自闭合规范化；
- 普通 Markdown、普通文本、`data-style` 和合法 MDX style 表达式保持不变；
- 首次创建和动态更新都会经过清理；
- 关键字参数会原样转交给 Viser；
- 清理函数幂等；
- AST 静态守卫禁止兼容层之外的 viewer 模块再出现直接 `add_markdown()` 或直接 `.content = ...`。

运行方式：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\viewer\test_markdown_compat.py
```

本次结果：针对性测试 `15 passed`；全量测试 `828 passed, 6 skipped`。

## 5. 手动验证路径

在真实 Viser 1.0.26 页面中完成以下回归：

1. 打开 GUI，确认没有 `Markdown Failed to Render`；
2. 选择 `Xsens_mocap · stand`，确认 Clip info 显示名称、帧数、FPS、Bones 与轴信息；
3. 切换到 Robot，选择 `G1 29dof · g1_29dof`；
4. 加载机器人，覆盖进度与机器人状态的动态 Markdown 更新；
5. 打开已有标定确认对话框，确认带换行的内容可正常渲染；
6. 检查上述操作之后没有新增浏览器错误。

上述路径已在 Viser 1.0.26 的真实页面完成，操作时间点之后的浏览器错误记录为 0。

## 6. 版本说明

- `uv.lock` 当前固定 Viser 1.0.26，但 `pyproject.toml` 写的是 `viser>=0.2`，不同安装方式仍可能解析到不同版本；
- Viser 1.1.0 仍使用相关 MDX 渲染路径，单纯升级到 1.1.0 不能作为本问题的修复；
- Viser 尚未发布的 main 分支已经更换 Markdown 管道，但不建议仅为此问题直接依赖 main；
- `style={{...}}` 是合法的 MDX/JSX 表达式，兼容层会保留它；
- Python 项目可以通过完整 Git commit SHA 锁定 VCS dependency；这与本次本地兼容修复互不冲突。

## 7. 修改文件

| 文件 | 作用 |
|---|---|
| `hhtools/viewer/markdown_compat.py` | Viser Markdown 创建与更新的统一兼容边界 |
| `hhtools/viewer/app.py` | 接入安全创建/更新，并修复 Clip info 的低对比度展示 |
| `tests/viewer/test_markdown_compat.py` | 动态路径、边界行为与防回归静态检查 |
| `G1_VISER_MARKDOWN_FIX.md` | 将原“待实施方案”修订为本修复记录 |

## 8. 参考

- [React error #62](https://react.dev/errors/62)
- [Viser 未发布 Markdown 管道变更 08c9378](https://github.com/viser-project/viser/commit/08c9378944ab1b70486499483b7ff4415c8fb54c)
- [MDX expressions](https://mdxjs.com/docs/what-is-mdx/#expressions)
- [pip VCS support](https://pip.pypa.io/en/latest/topics/vcs-support/)
