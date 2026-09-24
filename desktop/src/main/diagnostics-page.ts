function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

export function diagnosticsDataUrl(details: {
  title: string
  message: string
  stage: string
  logPath?: string
  pythonPath?: string
}): string {
  const rows = [
    ['Startup stage', details.stage],
    ['Python', details.pythonPath ?? 'Not resolved'],
    ['Log file', details.logPath ?? 'Not created']
  ]
    .map(
      ([label, value]) =>
        `<div class="row"><span>${escapeHtml(label)}</span><code>${escapeHtml(value)}</code></div>`
    )
    .join('')

  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>${escapeHtml(details.title)}</title>
<style>
body{margin:0;background:#f7f9fc;color:#0b1b38;font:15px/1.6 system-ui,sans-serif;display:grid;min-height:100vh;place-items:center}
main{width:min(680px,calc(100vw - 48px));background:#fff;border:1px solid #dbe4f2;padding:32px;border-radius:8px;box-shadow:0 12px 36px #183d6d16}
h1{font-size:24px;margin:0 0 10px}p{color:#536987;margin:0 0 24px}.row{display:grid;grid-template-columns:140px 1fr;gap:16px;padding:12px 0;border-top:1px solid #e7edf6}.row span{font-weight:600}.row code{overflow-wrap:anywhere}
</style></head><body><main><h1>${escapeHtml(details.title)}</h1><p>${escapeHtml(details.message)}</p>${rows}</main></body></html>`
  return `data:text/html;charset=UTF-8,${encodeURIComponent(html)}`
}
