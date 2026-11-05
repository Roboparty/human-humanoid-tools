# Job scheduling

**[中文](./job-scheduling_cn.md)**

Web jobs and Agent batches are unlimited by default. On a shared or memory-constrained GPU, you can enable FIFO admission control:

```bash
uv run hhtools web --max-running-jobs 1 --max-queued-jobs 32
```

## Limits

`0` means unlimited for every option.

| CLI option | Environment variable | Limits |
|------------|----------------------|--------|
| `--max-running-jobs` | `HHTOOLS_MAX_RUNNING_JOBS` | Concurrently running Web jobs |
| `--max-queued-jobs` | `HHTOOLS_MAX_QUEUED_JOBS` | Queue capacity; applies only when running concurrency is limited |
| `--max-batch-items` | `HHTOOLS_MAX_BATCH_ITEMS` | Items in an Agent batch |
| `--max-batch-total-frames` | `HHTOOLS_MAX_BATCH_TOTAL_FRAMES` | Total frames in an Agent batch |

The environment variables also work in the Electron sidecar and the MCP server.

The concurrency cap applies to scheduled Web jobs, not the optional Warp/Newton robot prewarm thread, so it is admission control rather than a process-wide GPU concurrency guarantee.

## Changing limits at runtime

Edit the limits under **Settings → Background-job scheduling** from local Web, Electron, or an SSH loopback tunnel. Ordinary remote-browser sessions are shown read-only until authenticated remote administration is implemented.

Saving hot-applies the limits without restarting Python or Electron:

- Lower running limits grandfather active jobs.
- Higher limits immediately promote FIFO waiters.

The backend persists the values in the platform user-config directory; `HHTOOLS_WEB_SETTINGS_PATH` selects another file. Explicit CLI or environment values remain startup overrides and win again on the next launch.
