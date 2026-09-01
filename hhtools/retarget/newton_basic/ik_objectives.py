# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from soma-retargeter (Apache-2.0).
# See https://github.com/NVIDIA/SOMA-Retargeter and the project root NOTICE.
"""Custom Newton IK objectives used by ``newton_basic`` pipeline.

Currently only :class:`IKSmoothJointFilter` — a soft penalty that pulls every
joint coordinate towards the midpoint of its configured limits.  Unlike the
stock :class:`newton.ik.IKObjectiveJointLimit` (a hinge at the limits) this
acts everywhere, giving the IK solver a stable preferred pose so the shoulder
/ waist rotations don't drift over a sequence.

The implementation is a direct Apache-2.0 port of
``soma_retargeter.pipelines.ik_objectives.IKSmoothJointFilter``; any future
rebalancing lives here.
"""

from __future__ import annotations

import numpy as np
import warp as wp
import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


__all__ = ["IKSmoothJointFilter"]


@wp.func
def _wp_smooth_joint_filter_func(
    x: wp.float32,
    lower_limit: wp.float32,
    upper_limit: wp.float32,
    padding_limit: wp.float32,
    m: wp.float32,
    p: wp.float32,
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += padding_limit - c
    upper_limit -= padding_limit + c
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit - x, x - upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    joint_limit_lower: wp.array1d(dtype=wp.float32),
    joint_limit_upper: wp.array1d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx]
    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5
        q = joint_q[problem, coord_idx]
        error = q - c
        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),
    coord_masks: wp.array1d(dtype=wp.float32),
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32),
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    mask = coord_masks[coord_idx]
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),
):
    out_weight[0] = in_value


class IKSmoothJointFilter(ik.IKObjective):
    """Smooth pull-to-midpoint joint limit objective.

    Args:
        joint_limit_lower: ``(n_dofs,)`` lower limit per DOF (warp array).
        joint_limit_upper: ``(n_dofs,)`` upper limit per DOF (warp array).
        weight: Scalar objective weight.  Can be updated at runtime via
            :meth:`set_weight`.
        coord_masks: Optional ``(n_coords,)`` mask in ``[0, 1]``: 0 disables
            the penalty for that coordinate (useful to leave e.g. the
            floating-base root untouched).  ``None`` means "all coords".
    """

    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if (
            self.coord_masks_np is not None
            and len(self.coord_masks_np) == model.joint_coord_count
        ):
            self.coord_masks = wp.array(
                self.coord_masks_np, dtype=wp.float32, device=self.device
            )
        if self.coord_masks is None:
            self.coord_masks = wp.ones(
                shape=model.joint_coord_count, dtype=wp.float32, device=self.device
            )

        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return
        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})
        _ = tape.gradients[dq_dof]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(
        self, body_q, joint_q, model, jacobian, joint_S_s, start_idx
    ):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
