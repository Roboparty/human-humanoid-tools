# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from soma-retargeter (Apache-2.0).
# See https://github.com/NVIDIA/soma-retargeter and the project root NOTICE.
"""Ground-plane collision IK objective for ``newton_basic``.

Port of ``soma_retargeter.pipelines.ik_collision_objective.IKObjectiveGroundCollision``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
import newton.ik as ik

from hhtools.retarget.newton_basic.ground_collision_bodies import (
    resolve_ground_collision_bodies,
)

__all__ = ["IKObjectiveGroundCollision", "resolve_ground_collision_bodies"]


@wp.func
def _capsule_min_z(
    body_tf: wp.transform,
    local_p0: wp.vec3,
    local_p1: wp.vec3,
    radius: wp.float32,
) -> wp.float32:
    w0 = wp.transform_point(body_tf, local_p0)
    w1 = wp.transform_point(body_tf, local_p1)
    return wp.min(w0[2], w1[2]) - radius


@wp.func
def _capsule_lowest_point(
    body_tf: wp.transform,
    local_p0: wp.vec3,
    local_p1: wp.vec3,
) -> wp.vec3:
    w0 = wp.transform_point(body_tf, local_p0)
    w1 = wp.transform_point(body_tf, local_p1)
    if w0[2] <= w1[2]:
        return w0
    return w1


@wp.kernel
def _ground_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_indices: wp.array1d(dtype=wp.int32),
    local_p0: wp.array1d(dtype=wp.vec3),
    local_p1: wp.array1d(dtype=wp.vec3),
    radius: wp.array1d(dtype=wp.float32),
    margin: wp.array1d(dtype=wp.float32),
    ground_z: wp.float32,
    weight: wp.array(dtype=wp.float32),
    start_idx: wp.int32,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, idx = wp.tid()
    bi = body_indices[idx]
    min_z = _capsule_min_z(body_q[problem, bi], local_p0[idx], local_p1[idx], radius[idx])
    threshold = ground_z + margin[idx]
    viol = wp.max(0.0, threshold - min_z)
    w = weight[0]
    residuals[problem, start_idx + idx] = w * viol


@wp.kernel
def _ground_jac_analytic(
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    body_indices: wp.array1d(dtype=wp.int32),
    local_p0: wp.array1d(dtype=wp.vec3),
    local_p1: wp.array1d(dtype=wp.vec3),
    radius: wp.array1d(dtype=wp.float32),
    margin: wp.array1d(dtype=wp.float32),
    ground_z: wp.float32,
    affects_dof: wp.array2d(dtype=wp.uint8),
    weight: wp.array(dtype=wp.float32),
    start_idx: wp.int32,
    n_dofs: wp.int32,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, idx = wp.tid()
    bi = body_indices[idx]
    tf = body_q[problem, bi]

    min_z = _capsule_min_z(tf, local_p0[idx], local_p1[idx], radius[idx])
    threshold = ground_z + margin[idx]
    if min_z >= threshold:
        return

    lowest = _capsule_lowest_point(tf, local_p0[idx], local_p1[idx])
    residual_row = start_idx + idx
    w = weight[0]

    for dof in range(n_dofs):
        if affects_dof[idx, dof] == wp.uint8(0):
            continue

        S = joint_S_s[problem, dof]
        v = wp.vec3(S[0], S[1], S[2])
        omega = wp.vec3(S[3], S[4], S[5])
        vel = v + wp.cross(omega, lowest)
        jacobian[problem, residual_row, dof] = -w * vel[2]


class IKObjectiveGroundCollision(ik.IKObjective):
    """Capsule-vs-ground-plane penalties (soma-compatible)."""

    def __init__(
        self,
        model,
        *,
        body_labels: list[str],
        ground_bodies: list[dict[str, Any]],
        weight: float = 1.0,
        ground_z: float = 0.0,
    ):
        super().__init__()
        resolved = resolve_ground_collision_bodies(body_labels, ground_bodies)
        self.n_bodies = len(resolved)
        self.weight = float(weight)
        self._weight_np = np.array([self.weight], dtype=np.float32)
        self.ground_z = float(ground_z)

        bi = np.empty(self.n_bodies, dtype=np.int32)
        p0 = np.zeros((self.n_bodies, 3), dtype=np.float32)
        p1 = np.zeros((self.n_bodies, 3), dtype=np.float32)
        ra = np.zeros(self.n_bodies, dtype=np.float32)
        mg = np.full(self.n_bodies, 0.02, dtype=np.float32)

        for i, entry in enumerate(resolved):
            bi[i] = body_labels.index(str(entry["body"]))
            cap = entry["capsule"]
            p0[i] = np.asarray(cap[0], dtype=np.float32)
            p1[i] = np.asarray(cap[1], dtype=np.float32)
            ra[i] = float(cap[2])
            mg[i] = float(entry.get("margin", 0.02))

        self._bi = bi
        self._p0 = p0
        self._p1 = p1
        self._ra = ra
        self._mg = mg

        joint_child_np = model.joint_child.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        n_dofs = model.joint_dof_count
        body_count = model.body_count

        body_to_joint = np.full(body_count, -1, np.int32)
        for j in range(model.joint_count):
            c = joint_child_np[j]
            if c != -1:
                body_to_joint[c] = j

        dof_to_joint = np.empty(n_dofs, dtype=np.int32)
        for j in range(len(joint_qd_start_np) - 1):
            dof_to_joint[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = j

        def _ancestor_dofs(body_idx: int) -> np.ndarray:
            mask = np.zeros(n_dofs, dtype=np.uint8)
            b = body_idx
            while b != -1:
                j = body_to_joint[b]
                if j == -1:
                    break
                anc = np.zeros(model.joint_count, dtype=bool)
                anc[j] = True
                mask[anc[dof_to_joint]] = 1
                b = joint_parent_np[j]
            return mask

        self._affects = np.zeros((self.n_bodies, n_dofs), dtype=np.uint8)
        for i in range(self.n_bodies):
            self._affects[i] = _ancestor_dofs(bi[i])

        self.d_bi = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        d = self.device
        self.d_bi = wp.array(self._bi, dtype=wp.int32, device=d)
        self.d_p0 = wp.array(self._p0, dtype=wp.vec3, device=d)
        self.d_p1 = wp.array(self._p1, dtype=wp.vec3, device=d)
        self.d_ra = wp.array(self._ra, dtype=wp.float32, device=d)
        self.d_mg = wp.array(self._mg, dtype=wp.float32, device=d)
        self.d_affects = wp.array(self._affects, dtype=wp.uint8, device=d)
        self.d_weight = wp.array(self._weight_np, dtype=wp.float32, device=d)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_bodies

    def set_weight(self, value: float):
        self.weight = float(value)
        self._weight_np[0] = self.weight
        if getattr(self, "d_weight", None) is not None:
            self.d_weight.assign(self._weight_np)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = body_q.shape[0]
        wp.launch(
            _ground_residuals,
            dim=[count, self.n_bodies],
            inputs=[
                body_q,
                self.d_bi,
                self.d_p0,
                self.d_p1,
                self.d_ra,
                self.d_mg,
                self.ground_z,
                self.d_weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        pass

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        n_dofs = model.joint_dof_count
        count = body_q.shape[0]
        wp.launch(
            _ground_jac_analytic,
            dim=[count, self.n_bodies],
            inputs=[
                body_q,
                joint_S_s,
                self.d_bi,
                self.d_p0,
                self.d_p1,
                self.d_ra,
                self.d_mg,
                self.ground_z,
                self.d_affects,
                self.d_weight,
                start_idx,
                n_dofs,
            ],
            outputs=[jacobian],
            device=self.device,
        )
