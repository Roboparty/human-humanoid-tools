# Enable GitHub Pages

After pushing the `docs/` folder to `main`:

1. Open https://github.com/Roboparty/human-humanoid-tools/settings/pages
2. **Build and deployment** → Source: **Deploy from a branch**
3. Branch: **main** · Folder: **`/docs`**（不是 `/ (root)`）
4. Save — the site will be live at https://roboparty.github.io/human-humanoid-tools/

> **常见错误**：若 Source 选的是 `/ (root)`，首页会渲染仓库根目录的 `README.md`，自定义宣传页只会出现在 `/docs/index.html`，且样式/视频路径容易不对。务必选 **`/docs`**。

## Local preview

```bash
cd docs && python3 -m http.server 8080
# open http://127.0.0.1:8080
```

## Add demo videos

Drop MP4 files into `assets/videos/` (see `assets/videos/README.md`). Refresh the page — placeholders disappear automatically when files exist.
