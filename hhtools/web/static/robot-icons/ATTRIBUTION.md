# Robot Library icon sources

These 128 px WebP thumbnails identify the six robot presets curated by
HHTools. They are deterministic zero-pose renders made from the official URDF
and mesh assets listed below, rather than vendor logos or promotional photos.
The reproducible renderer is `scripts/render_robot_library_icons.py`.

For every thumbnail, HHTools contributors made the following changes on
2026-08-28: assembled the upstream meshes through the upstream URDF, rendered a
fixed orthographic three-quarter projection, applied depth shading and a
coloured identification tile, resized the result to 128 x 128 pixels, and
encoded it as a lossless WebP. Each thumbnail remains available under the
license shown beside it; those licenses do not change the Apache-2.0 license of
the surrounding HHTools application.

- **Unitree G1** (`unitree-g1.webp`) — rendered from the official 29-DoF model
  in [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/7d6075f7f58588b189b940130e3edab3c839b2df/robots/g1_description),
  under the bundled [BSD-3-Clause license](licenses/unitree-g1-BSD-3-Clause.txt).
- **ROBOTO_ORIGIN (RPO)** (`roboto-origin.webp`) — rendered from
  `urdf/rpo.urdf` and `meshes/` in
  [`Roboparty/rpo_description`](https://github.com/Roboparty/rpo_description/tree/37aac9ca665e92731444a1618320078e7ba21569),
  under the bundled
  [CERN-OHL-W-2.0 license](licenses/roboto-origin-CERN-OHL-W-2.0.txt).
- **AgiBot X2 Ultra** (`agibot-x2.webp`) — rendered from the official v1.4 X2
  model in
  [`AgibotTech/agibot_x2_urdf`](https://github.com/AgibotTech/agibot_x2_urdf/tree/77f43eb0904dae4c48ccd9154fee824f8ffd4d38/X2_URDF-v1.4.0),
  under the bundled [Mulan PSL v2](licenses/agibot-x2-MulanPSL-2.0.txt).
- **Asimov 1** (`asimov-1.webp`) — rendered from `sim-model/urdf/asimov_1.urdf`
  and `sim-model/assets/meshes/` in
  [`menloresearch/asimov-1`](https://github.com/menloresearch/asimov-1/tree/b8420ffe99159065152aa1321a03147c0962f251/sim-model),
  under the bundled
  [CERN-OHL-S-2.0 license](licenses/asimov-1-CERN-OHL-S-2.0.txt).
- **Fourier GR-2** (`fourier-gr2.webp`) — rendered from
  `GRX/GR2/gr2v3_8_7/basic_urdf/` in
  [`FFTAI/Wiki-GRx-Models`](https://github.com/FFTAI/Wiki-GRx-Models/tree/7d96c758f048fe1bf92b3258864d94771ae0c093/GRX/GR2/gr2v3_8_7/basic_urdf),
  under the bundled [GPL-3.0 license](licenses/fourier-gr2-GPL-3.0.txt).
- **Berkeley Humanoid Lite** (`berkeley-humanoid-lite.webp`) — rendered from
  the official description in
  [`HybridRobotics/Berkeley-Humanoid-Lite-Assets`](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets/tree/fc90fedd008b1e56a22e3c5221548d6b24f49707/data/robots/berkeley_humanoid/berkeley_humanoid_lite),
  under the bundled
  [CC BY-SA 4.0 license](licenses/berkeley-humanoid-lite-CC-BY-SA-4.0.txt).

Names and trademarks belong to their respective owners. Inclusion identifies
compatible models and does not imply endorsement. The unmodified HHTools robot
mark remains the icon for every user-imported model.
