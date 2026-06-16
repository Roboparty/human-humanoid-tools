# Changelog

All notable changes to this project will be documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project scaffold: `pyproject.toml`, Apache-2.0 `LICENSE`, README (bilingual), contribution
  guidelines, GitHub Actions CI, and the `hhtools` Python package skeleton.
- Core data model: `Hierarchy`, `Skeleton`, `Motion`, `AnimationBuffer`, and a vectorised math
  layer (quaternion, rotation, transform, vector utilities) in `hhtools.core`.
- Unified internal NPZ schema at `hhtools.io.npz` together with a self-written Apache-2.0 BVH
  parser and exporter at `hhtools.io.bvh`.
- `hhtools convert` CLI entry point for headless BVH/GLB to NPZ batch conversion.
- Minimal Viser + FastAPI viewer (`hhtools ui`) that loads an NPZ and draws a playable skeleton.
