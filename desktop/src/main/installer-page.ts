function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

export function installerDataUrl(details: {
  version: string
  reason: string
  allowSystemInstall?: boolean
}): string {
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Set up Human-Humanoid Tools</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#191d24;color:#f5f7fb}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 50% -15%,#204d7a55,transparent 45%),#191d24}
main{width:min(780px,calc(100vw - 40px));margin:0 auto;padding:42px 0 36px}.hero{display:flex;align-items:center;gap:18px;margin-bottom:24px}
.mark{position:relative;display:grid;place-items:center;width:62px;height:62px;border-radius:18px;background:#4fa4ff1c;border:1px solid #5aabff55;box-shadow:0 12px 30px #0006}
.mark:before,.mark:after{content:"";position:absolute;border-radius:50%;border:2px solid #55aaff}.mark:before{width:18px;height:18px}.mark:after{width:42px;height:42px;border-color:#55aaff55;animation:orbit 2.4s linear infinite}
.mark span{width:7px;height:7px;background:#6ff0c4;border-radius:50%;transform:translate(19px);box-shadow:0 0 15px #6ff0c4;animation:pulse 1.2s ease-in-out infinite alternate}
@keyframes orbit{to{transform:rotate(360deg)}}@keyframes pulse{to{opacity:.35;transform:translate(19px) scale(.72)}}
h1{font-size:28px;line-height:1.15;margin:0 0 7px}p{margin:0;color:#aab5c7}.version{color:#62afff;font-size:13px;font-weight:650}
.panel{background:#202630;border:1px solid #354052;border-radius:12px;padding:22px;box-shadow:0 16px 50px #0004}.reason{font-size:13px;padding:11px 13px;border-radius:8px;background:#151a21;color:#9eacc0;margin-bottom:18px}
h2{font-size:15px;margin:0 0 12px}.modes{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mode{display:block;border:1px solid #3c485a;border-radius:10px;padding:15px;cursor:pointer;transition:.15s;background:#1b2028}
.mode:has(input:checked){border-color:#55aaff;background:#173451;box-shadow:0 0 0 1px #55aaff44}.mode input{accent-color:#55aaff;margin:0 8px 0 0}.mode strong{font-size:14px}.mode small{display:block;color:#9ba9bd;margin:8px 0 0 24px;line-height:1.45}
.components{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:18px 0}.component{background:#181d24;border-radius:8px;padding:10px 12px;font-size:12px;color:#b8c3d2}.component b{display:block;color:#eff4fb;margin-bottom:2px}
.license{display:flex;gap:9px;align-items:flex-start;font-size:12px;color:#aeb9c9;padding:13px 0;border-top:1px solid #333d4b}.license input{margin-top:3px;accent-color:#55aaff}.license a{color:#61afff}
.progress{display:none;margin-top:16px;border-top:1px solid #333d4b;padding-top:16px}.progress[data-visible="true"]{display:block}.progress-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.phase{font-size:13px;font-weight:650}.spinner{width:16px;height:16px;border:2px solid #4c596a;border-top-color:#59aeff;border-radius:50%;animation:orbit .8s linear infinite}.bar{height:4px;border-radius:4px;background:#303a48;overflow:hidden;margin:11px 0}.bar:after{content:"";display:block;width:35%;height:100%;background:linear-gradient(90deg,transparent,#59aeff,#72e9c1,transparent);animation:scan 1.5s ease-in-out infinite}@keyframes scan{from{transform:translateX(-100%)}to{transform:translateX(300%)}}
.log{height:100px;overflow:auto;white-space:pre-wrap;background:#12161c;color:#8fa0b5;border-radius:7px;padding:9px;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.actions{display:flex;justify-content:flex-end;gap:10px;margin-top:18px}
button{border:1px solid #455166;background:#252c37;color:#eaf0f8;border-radius:7px;padding:9px 16px;font-weight:650;cursor:pointer}button.primary{background:#4fa4f7;border-color:#4fa4f7;color:#071525}button:disabled{opacity:.45;cursor:not-allowed}.error{color:#ff8b8b}.success{color:#70e8bf}
@media(max-width:620px){.modes,.components{grid-template-columns:1fr}main{padding-top:24px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important}}
</style></head><body><main>
<header class="hero"><div class="mark" aria-hidden="true"><span></span></div><div><div class="version">HHTools ${escapeHtml(details.version)}</div><h1 data-copy="title">Set up the local runtime</h1><p data-copy="subtitle">Install the Python environment required by the desktop application.</p></div></header>
<section class="panel"><div class="reason" data-copy="reason" title="${escapeHtml(details.reason)}">The local runtime for this app version is not installed yet.</div><h2 data-copy="scope">Installation scope</h2>
<div class="modes">
<label class="mode"><input type="radio" name="mode" value="user" checked><strong data-copy="userTitle">Current user · Recommended</strong><small data-copy="userDetail">Installs under your user data directory. No administrator password is needed.</small></label>
${details.allowSystemInstall === false ? '' : '<label class="mode"><input type="radio" name="mode" value="system"><strong data-copy="systemTitle">All users · Administrator</strong><small data-copy="systemDetail">Uses the operating system authentication dialog. HHTools never receives your password.</small></label>'}
</div>
<div class="components"><div class="component"><b data-copy="runtime">Python runtime</b><span data-copy="runtimeDetail">Isolated and versioned</span></div><div class="component"><b>HHTools</b><span data-copy="deps">Web, retarget and Agent dependencies</span></div><div class="component"><b data-copy="assets">Built-in assets</b><span data-copy="assetsDetail">30 motions and 6 robots</span></div></div>
<label class="license"><input id="license" type="checkbox"><span data-copy="license">I agree to install HHTools and its runtime dependencies. I understand that third-party components remain subject to their own licenses, and optional GVHMR and SMPL-family weights are not installed automatically.</span></label>
<div id="progress" class="progress" data-visible="false" aria-live="polite"><div class="progress-head"><span id="phase" class="phase">Ready</span><span id="spinner" class="spinner" aria-hidden="true"></span></div><div class="bar"></div><div id="log" class="log"></div></div>
<div class="actions"><button id="cancel" hidden data-copy="cancel">Cancel</button><button id="install" class="primary" disabled data-copy="install">Install and start</button></div></section>
</main><script>
const copy={zh:{title:'设置本地运行环境',subtitle:'安装桌面应用所需的 Python 环境。',reason:'尚未安装与当前应用版本匹配的本地运行环境。',scope:'安装范围',userTitle:'仅当前用户 · 推荐',userDetail:'安装到用户数据目录，不需要管理员密码。',systemTitle:'所有用户 · 管理员',systemDetail:'使用操作系统认证窗口；HHTools 不会接触你的密码。',runtime:'Python 运行环境',runtimeDetail:'隔离、按版本管理',deps:'Web、重定向与 Agent 全套依赖',assets:'内置资源',assetsDetail:'30 条动作与 6 个机器人',license:'我同意安装 HHTools 及其运行依赖；我了解第三方组件仍受各自许可约束，且可选的 GVHMR 与 SMPL 系权重不会自动安装。',cancel:'取消',install:'安装并启动'},en:{}}
if(navigator.language.toLowerCase().startsWith('zh'))for(const node of document.querySelectorAll('[data-copy]')){const value=copy.zh[node.dataset.copy];if(value)node.textContent=value}
const api=window.hhtoolsInstaller,license=document.getElementById('license'),install=document.getElementById('install'),cancel=document.getElementById('cancel'),progress=document.getElementById('progress'),phase=document.getElementById('phase'),log=document.getElementById('log'),spinner=document.getElementById('spinner');let running=false
license.addEventListener('change',()=>{install.disabled=!license.checked||running})
api.onProgress(update=>{progress.dataset.visible='true';phase.textContent=update.message;phase.className='phase '+(update.phase==='failed'?'error':update.phase==='completed'?'success':'');if(update.detail){log.textContent=(log.textContent+'\\n'+update.detail).trim().split('\\n').slice(-80).join('\\n');log.scrollTop=log.scrollHeight}if(['failed','cancelled'].includes(update.phase)){running=false;install.disabled=!license.checked;cancel.hidden=true;spinner.hidden=true}else if(update.phase==='completed'){cancel.hidden=true;spinner.hidden=true}})
install.addEventListener('click',async()=>{if(running||!license.checked)return;running=true;install.disabled=true;cancel.hidden=false;spinner.hidden=false;log.textContent='';const mode=document.querySelector('input[name="mode"]:checked').value;try{await api.start(mode)}catch(error){progress.dataset.visible='true';phase.textContent=String(error);phase.className='phase error';running=false;install.disabled=false;cancel.hidden=true;spinner.hidden=true}})
cancel.addEventListener('click',()=>api.cancel())
</script></body></html>`
  return `data:text/html;charset=UTF-8,${encodeURIComponent(html)}`
}
