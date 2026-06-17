# Enable GitHub Pages

After pushing the `docs/` folder to `main`:

1. Open https://github.com/jaggerShen/human-humanoid-tools/settings/pages
2. **Build and deployment** → Source: **Deploy from a branch**
3. Branch: **main** · Folder: **/docs**
4. Save — the site will be live at https://jaggerShen.github.io/human-humanoid-tools/

## Local preview

```bash
cd docs && python3 -m http.server 8080
# open http://127.0.0.1:8080
```

## Add demo videos

Drop MP4 files into `assets/videos/` (see `assets/videos/README.md`). Refresh the page — placeholders disappear automatically when files exist.
