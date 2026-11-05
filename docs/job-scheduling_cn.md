# 任务调度

**[English](job-scheduling.md)**

Web 任务与 Agent 批处理默认都不限制。在共享或显存紧张的 GPU 上，可以启用 FIFO 准入控制：

```bash
uv run hhtools web --max-running-jobs 1 --max-queued-jobs 32
```

## 限制项

所有选项的 `0` 都表示不限。

| CLI 选项 | 环境变量 | 限制对象 |
|----------|----------|----------|
| `--max-running-jobs` | `HHTOOLS_MAX_RUNNING_JOBS` | 同时运行的 Web 任务数 |
| `--max-queued-jobs` | `HHTOOLS_MAX_QUEUED_JOBS` | 等待队列容量；仅在运行并发受限时生效 |
| `--max-batch-items` | `HHTOOLS_MAX_BATCH_ITEMS` | 单个 Agent 批处理的条目数 |
| `--max-batch-total-frames` | `HHTOOLS_MAX_BATCH_TOTAL_FRAMES` | 单个 Agent 批处理的总帧数 |

这些环境变量在 Electron sidecar 和 MCP server 中同样生效。

并发上限只约束调度器管理的 Web 任务，不包括可选的 Warp/Newton 机器人预热线程，因此它是任务准入控制，而不是整个进程的 GPU 并发保证。

## 运行时修改

可以从本机 Web、Electron 或 SSH 本地回环隧道，在 **设置 → 后台任务调度** 中修改。在实现远程管理鉴权之前，普通远程浏览器会显示为只读。

保存后会热更新，无需重启 Python 或 Electron：

- 降低运行上限不会中断正在运行的任务。
- 提高上限会立即按 FIFO 顺序启动等待中的任务。

后端会把配置写入平台的用户配置目录，也可以用 `HHTOOLS_WEB_SETTINGS_PATH` 指定其他文件。显式的 CLI 或环境变量仍是启动时的覆盖项，下次启动时会再次覆盖界面中保存的值。
