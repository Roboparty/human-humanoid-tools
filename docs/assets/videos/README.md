# Demo videos for the project page

Place MP4 files in this directory. The site hides the placeholder once the `<video>` element loads metadata.

**Web playback requires `-movflags +faststart`** (moov atom at file start). Without it, browsers must download the entire file before play starts.

**Recommended encoding** (keep each file under ~50 MB for GitHub):

```bash
ffmpeg -i raw.mp4 -an -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p \
  -movflags +faststart highlight-01-fast-retarget.mp4
```

To strip audio from an existing file without re-encoding video:

```bash
ffmpeg -i raw.mp4 -an -c:v copy -movflags +faststart output.mp4
```

Optional posters (JPEG) go in `../posters/` with matching names.

## Recording checklist

| File | Section | Suggested content | Duration |
|------|---------|-------------------|----------|
| `highlight-01-fast-retarget.mp4` | Fast Retarget | LAFAN clip → G1 → Retarget → play comparison | 30–60 s |
| `highlight-02-any-motion-1.mp4` | Any Motion (1/3) | BVH / SMPL mimic clip load & preview | 20–40 s |
| `highlight-02-any-motion-2.mp4` | Any Motion (2/3) | OMOMO / intermimic with objects | 20–40 s |
| `highlight-02-any-motion-3.mp4` | Any Motion (3/3) | meshmimic parkour / terrain | 20–40 s |
| `highlight-03-any-urdf-1.mp4` | Any URDF (1/2) | Drag URDF + meshes → auto scaffold | 30–45 s |
| `highlight-03-any-urdf-2.mp4` | Any URDF (2/2) | First calibration & validate | 30–45 s |
| `highlight-04-batch-retarget.mp4` | Batch Retarget | Multi-select library → batch run → ZIP export | 45–60 s |
| `highlight-05-r2r.mp4` | Robot → Robot | Source robot CSV → target URDF → export | ~45 s |
| `highlight-06-dataset-viz.mp4` | Dataset Analysis | Scatter brush, tag filter, subset export | ~60 s |
| `highlight-08-method.mp4` | Method (optional) | Solver / backends overview | 30–60 s |

## Recording on Linux (NVIDIA GPU)

**OBS Studio** (recommended): 1920×1080 @ 60 fps, encoder **NVENC H.264**, window capture on the browser.

```bash
sudo apt install obs-studio
```

**ffmpeg** (scriptable):

```bash
ffmpeg -f x11grab -framerate 60 -video_size 1920x1080 -i :0.0+0,0 \
  -c:v h264_nvenc -preset p5 -cq 20 -pix_fmt yuv420p output.mp4
```

Wayland: use OBS PipeWire capture or `wf-recorder`.

## Tips

1. Browser zoom 125–150%, hide bookmarks bar.
2. Use light theme to match the Web UI.
3. One short flow per highlight; use built-in `assets/motions/` demos.
4. Wait for retarget progress to finish before cutting.
5. Do not expose local paths or secrets in the recording.
