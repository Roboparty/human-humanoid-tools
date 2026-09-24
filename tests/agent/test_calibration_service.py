from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hhtools.contracts import (
    AssetRegistrationRequest,
    CalibrationPreviewRequest,
    CalibrationProposalRequest,
    CalibrationSaveRequest,
    CalibrationStatusRequest,
    CalibrationValidationRequest,
    CalibrationVisualReview,
)
from hhtools.retarget.calibration import RobotRetargetCalibration, save_calibration
from hhtools.robot.loader import load_robot
from hhtools.robot.registry import preset_from_dir
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.calibration import CalibrationService, CalibrationServiceError
from hhtools.services.calibration_candidates import CalibrationCandidateStore


def _write_robot(root: Path) -> None:
    root.mkdir(parents=True)
    joints: list[str] = []
    links = ["pelvis", "spine_link", "torso_link", "head_link"]
    xml = [
        '<robot name="test_robot">',
        '<link name="pelvis"/>',
        '<link name="spine_link"/>',
        '<joint name="waist_joint" type="revolute">',
        '<parent link="pelvis"/><child link="spine_link"/>',
        '<origin xyz="0 0 0.15"/><axis xyz="0 1 0"/>',
        '<limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>',
        "</joint>",
        '<link name="torso_link"/>',
        '<joint name="chest_fixed" type="fixed">',
        '<parent link="spine_link"/><child link="torso_link"/>',
        '<origin xyz="0 0 0.2"/>',
        "</joint>",
        '<link name="head_link"/>',
        '<joint name="head_fixed" type="fixed">',
        '<parent link="torso_link"/><child link="head_link"/>',
        '<origin xyz="0 0 0.25"/>',
        "</joint>",
    ]
    joints.append("waist_joint")
    for side, lateral, roll_sign in (("left", 0.18, 1.0), ("right", -0.18, -1.0)):
        upper = f"{side}_upper_arm_link"
        elbow = f"{side}_elbow_link"
        wrist = f"{side}_wrist_link"
        thigh = f"{side}_thigh_link"
        knee = f"{side}_knee_link"
        ankle = f"{side}_ankle_link"
        links.extend((upper, elbow, wrist, thigh, knee, ankle))
        shoulder_joint = f"{side}_shoulder_roll_joint"
        elbow_joint = f"{side}_elbow_joint"
        hip_joint = f"{side}_hip_joint"
        knee_joint = f"{side}_knee_joint"
        ankle_joint = f"{side}_ankle_joint"
        joints.extend((shoulder_joint, elbow_joint, hip_joint, knee_joint, ankle_joint))
        shoulder_lower, shoulder_upper = (-0.1, 2.2) if roll_sign > 0 else (-2.2, 0.1)
        xml.extend(
            [
                f'<link name="{upper}"/>',
                f'<joint name="{shoulder_joint}" type="revolute">',
                f'<parent link="torso_link"/><child link="{upper}"/>',
                f'<origin xyz="0 {lateral} 0"/><axis xyz="1 0 0"/>',
                (
                    f'<limit lower="{shoulder_lower}" upper="{shoulder_upper}" '
                    'effort="10" velocity="2"/>'
                ),
                "</joint>",
                f'<link name="{elbow}"/>',
                f'<joint name="{elbow_joint}" type="revolute">',
                f'<parent link="{upper}"/><child link="{elbow}"/>',
                '<origin xyz="0 0 -0.3"/><axis xyz="1 0 0"/>',
                '<limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>',
                "</joint>",
                f'<link name="{wrist}"/>',
                f'<joint name="{side}_wrist_fixed" type="fixed">',
                f'<parent link="{elbow}"/><child link="{wrist}"/>',
                '<origin xyz="0 0 -0.28"/>',
                "</joint>",
                f'<link name="{thigh}"/>',
                f'<joint name="{hip_joint}" type="revolute">',
                f'<parent link="pelvis"/><child link="{thigh}"/>',
                f'<origin xyz="0 {0.1 if lateral > 0 else -0.1} -0.05"/><axis xyz="0 1 0"/>',
                '<limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>',
                "</joint>",
                f'<link name="{knee}"/>',
                f'<joint name="{knee_joint}" type="revolute">',
                f'<parent link="{thigh}"/><child link="{knee}"/>',
                '<origin xyz="0 0 -0.38"/><axis xyz="0 1 0"/>',
                '<limit lower="-0.2" upper="2.4" effort="10" velocity="2"/>',
                "</joint>",
                f'<link name="{ankle}"/>',
                f'<joint name="{ankle_joint}" type="revolute">',
                f'<parent link="{knee}"/><child link="{ankle}"/>',
                '<origin xyz="0 0 -0.38"/><axis xyz="0 1 0"/>',
                '<limit lower="-0.8" upper="0.8" effort="10" velocity="2"/>',
                "</joint>",
            ]
        )
    xml.append("</robot>")
    (root / "robot.urdf").write_text("\n".join(xml), encoding="utf-8")
    (root / "robot.yaml").write_text(
        "name: test_robot\n"
        "display_name: Test Robot\n"
        "urdf: robot.urdf\n"
        f"dof_order: [{', '.join(joints)}]\n"
        "ik_map:\n"
        "  hips: pelvis\n"
        "  spine: spine_link\n"
        "  chest: torso_link\n"
        "  head: head_link\n"
        "  left_shoulder: left_upper_arm_link\n"
        "  left_elbow: left_elbow_link\n"
        "  left_wrist: left_wrist_link\n"
        "  right_shoulder: right_upper_arm_link\n"
        "  right_elbow: right_elbow_link\n"
        "  right_wrist: right_wrist_link\n"
        "  left_hip: left_thigh_link\n"
        "  left_knee: left_knee_link\n"
        "  left_ankle: left_ankle_link\n"
        "  right_hip: right_thigh_link\n"
        "  right_knee: right_knee_link\n"
        "  right_ankle: right_ankle_link\n",
        encoding="utf-8",
    )


def _service(
    tmp_path: Path,
    *,
    shared_robot_root: bool = False,
    bundled_reference: str | None = None,
) -> tuple[CalibrationService, str, Path]:
    robots = tmp_path / "robots"
    robot = robots / "test_robot"
    _write_robot(robot)
    user_root = robots if shared_robot_root else tmp_path / "user-robots"
    user_root.mkdir(exist_ok=True)
    if bundled_reference is not None:
        save_calibration(
            RobotRetargetCalibration(
                robot="test_robot",
                reference=bundled_reference,
                calibrated_joint_q={"waist_joint": 0.0},
            ),
            robot / f"retarget_calibration_{bundled_reference}.yaml",
        )
    assets = AgentAssetService(AssetRegistry(tmp_path / "agent-state", {"robots": robots}))
    bundle = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path="test_robot",
            kind="robot_bundle",
        )
    )
    preset = preset_from_dir(robot)

    def materialize(_robot_id: str, _asset_id: str):
        return load_robot(preset_from_dir(robot), compile_mjcf=False)

    service = CalibrationService(
        assets,
        CalibrationCandidateStore(tmp_path / "agent-state"),
        robot_provider=lambda: [preset],
        materialize_robot=materialize,
        release_robot=lambda _model: None,
        user_robot_root=user_root,
        request_id_provider=lambda: "req_calibration_test",
    )
    return service, bundle.asset_id, user_root


@pytest.mark.parametrize("reference", ["smpl", "smplx", "xsens_mocap", "lafan_bvh", "mocap_bvh"])
@pytest.mark.parametrize("replace_existing", [False, True])
def test_library_calibration_save_keeps_asset_identity_and_survives_restart(
    tmp_path: Path,
    reference: str,
    replace_existing: bool,
) -> None:
    service, asset_id, user_root = _service(
        tmp_path,
        shared_robot_root=True,
        bundled_reference=reference if replace_existing else None,
    )
    robot = user_root / "test_robot"
    before = {p.name: p.read_bytes() for p in robot.iterdir() if p.is_file()}
    identity = CalibrationStatusRequest(
        robot_id="test_robot",
        robot_asset_id=asset_id,
        reference=reference,
    )
    proposal = service.propose(CalibrationProposalRequest(**identity.model_dump()))
    request = CalibrationSaveRequest(
        candidate_id=proposal.candidate.candidate_id,
        save_mode="validated_silent",
    )
    receipt = service.save(request)
    saved = (
        user_root
        / ".calibration-overlays"
        / "test_robot"
        / f"retarget_calibration_{reference}.yaml"
    )
    saved_mtime = saved.stat().st_mtime_ns
    restarted = CalibrationService(
        AgentAssetService(AssetRegistry(tmp_path / "agent-state", {"robots": user_root})),
        CalibrationCandidateStore(tmp_path / "agent-state"),
        robot_provider=service._robot_provider,
        materialize_robot=service._materialize_robot,
        release_robot=lambda _model: None,
        user_robot_root=user_root,
    )
    assert restarted.save(request) == receipt
    assert saved.stat().st_mtime_ns == saved_mtime
    status = restarted.status(identity)
    assert status.state.value == "valid"
    assert status.robot_asset_id == asset_id
    assert status.calibration_digest == receipt.calibration_digest
    assert receipt.previous_calibration_archived is replace_existing
    assert {p.name: p.read_bytes() for p in robot.iterdir() if p.is_file()} == before
    assert (
        restarted._asset_service.register(
            AssetRegistrationRequest(
                root_id="robots",
                relative_path="test_robot",
                kind="robot_bundle",
            )
        ).asset_id
        == asset_id
    )
    saved.write_text(saved.read_text() + "\n# external edit\n")
    # A digest change is reflected in status and cannot reuse a previous plan.
    assert restarted.status(identity).calibration_digest != receipt.calibration_digest


def test_proposal_preview_and_validated_silent_save_round_trip(tmp_path: Path) -> None:
    service, robot_asset_id, user_root = _service(tmp_path)
    status_request = CalibrationStatusRequest(
        robot_id="test_robot",
        robot_asset_id=robot_asset_id,
        reference="smpl",
    )

    before = service.status(status_request)
    proposal = service.propose(
        CalibrationProposalRequest(
            robot_id="test_robot",
            robot_asset_id=robot_asset_id,
            reference="smpl",
        )
    )
    validation = service.validate(
        CalibrationValidationRequest(candidate_id=proposal.candidate.candidate_id)
    )
    preview, image = service.preview(
        CalibrationPreviewRequest(candidate_id=proposal.candidate.candidate_id)
    )
    receipt = service.save(
        CalibrationSaveRequest(
            candidate_id=proposal.candidate.candidate_id,
            save_mode="gpt_vision_silent",
            visual_review=CalibrationVisualReview(
                reviewer="gpt_vision",
                verdict="pass",
                model_hint="gpt-test",
                summary="Front and side landmark overlays are aligned.",
            ),
        )
    )
    replayed = service.save(
        CalibrationSaveRequest(
            candidate_id=proposal.candidate.candidate_id,
            save_mode="gpt_vision_silent",
            visual_review=CalibrationVisualReview(
                reviewer="gpt_vision",
                verdict="pass",
                model_hint="gpt-test",
                summary="Front and side landmark overlays are aligned.",
            ),
        )
    )
    after = service.status(status_request)

    assert before.state.value == "missing"
    assert before.joint_count == len(before.joint_limits) == 11
    assert proposal.validation.valid is True
    assert validation.valid is True
    assert validation.edge_errors_deg["left_upper_arm"] <= 20.0
    assert validation.edge_errors_deg["right_upper_arm"] <= 20.0
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert preview.sha256 == hashlib.sha256(image).hexdigest()
    assert receipt.saved is True
    assert replayed == receipt
    assert receipt.visual_review is not None
    assert receipt.visual_review.reviewer == "gpt_vision"
    saved = user_root / "test_robot" / "retarget_calibration_smpl.yaml"
    assert saved.is_file()
    assert after.state.value == "valid"
    assert after.calibration_id == receipt.calibration_id
    assert after.source == "user_calibration"


def test_locked_zero_pose_cannot_be_silently_saved(tmp_path: Path) -> None:
    service, robot_asset_id, _user_root = _service(tmp_path)
    joint_names = [
        "left_shoulder_roll_joint",
        "left_elbow_joint",
        "right_shoulder_roll_joint",
        "right_elbow_joint",
    ]
    proposal = service.propose(
        CalibrationProposalRequest(
            robot_id="test_robot",
            robot_asset_id=robot_asset_id,
            reference="smpl",
            joint_q_overrides={name: 0.0 for name in joint_names},
            locked_joints=joint_names,
        )
    )

    assert proposal.validation.valid is False
    with pytest.raises(CalibrationServiceError) as raised:
        service.save(
            CalibrationSaveRequest(
                candidate_id=proposal.candidate.candidate_id,
                save_mode="validated_silent",
            )
        )
    assert raised.value.code == "CALIBRATION_VALIDATION_FAILED"


def test_save_does_not_record_valid_evidence_for_replaced_bytes(tmp_path: Path, monkeypatch):
    from hhtools.retarget import calibration as storage
    from hhtools.services.calibration_validation import calibration_validation_identity

    service, asset_id, _ = _service(tmp_path)
    proposal = service.propose(
        CalibrationProposalRequest(
            robot_id="test_robot",
            robot_asset_id=asset_id,
            reference="smpl",
        )
    )
    original_save = storage.save_calibration_for_preset
    replaced = []

    def replace_after_save(*args, **kwargs):
        path = original_save(*args, **kwargs)
        calibration = storage.load_calibration(path)
        calibration.calibrated_joint_q["waist_joint"] = 0.4
        storage.save_calibration(calibration, path)
        replaced.append(path)
        return path

    monkeypatch.setattr(storage, "save_calibration_for_preset", replace_after_save)
    with pytest.raises(CalibrationServiceError) as raised:
        service.save(
            CalibrationSaveRequest(
                candidate_id=proposal.candidate.candidate_id,
                save_mode="validated_silent",
            )
        )
    assert raised.value.code == "CALIBRATION_SAVE_FAILED"
    identity = calibration_validation_identity(
        robot_id="test_robot",
        robot_asset_id=asset_id,
        reference="smpl",
        calibration_digest=hashlib.sha256(replaced[0].read_bytes()).hexdigest(),
    )
    assert service._candidate_store.validation_store.get(identity) is None


def test_visual_revision_creates_a_new_candidate_and_preserves_locked_overrides(
    tmp_path: Path,
) -> None:
    service, robot_asset_id, _user_root = _service(tmp_path)
    identity = {
        "robot_id": "test_robot",
        "robot_asset_id": robot_asset_id,
        "reference": "smpl",
    }
    first = service.propose(CalibrationProposalRequest(**identity))

    revised = service.propose(
        CalibrationProposalRequest(
            **identity,
            base_candidate_id=first.candidate.candidate_id,
            joint_q_overrides={"waist_joint": 0.1},
            locked_joints=["waist_joint"],
        )
    )

    assert revised.candidate.candidate_id != first.candidate.candidate_id
    assert revised.candidate.parent_candidate_id == first.candidate.candidate_id
    assert revised.candidate.joint_q["waist_joint"] == pytest.approx(0.1)
    assert revised.candidate.locked_joints == ["waist_joint"]
    assert service.validate(
        CalibrationValidationRequest(candidate_id=revised.candidate.candidate_id)
    ).valid


def test_status_marks_an_existing_but_arm_down_t_pose_calibration_invalid(
    tmp_path: Path,
) -> None:
    service, robot_asset_id, user_root = _service(tmp_path)
    target = user_root / "test_robot" / "retarget_calibration_smpl.yaml"
    save_calibration(
        RobotRetargetCalibration(
            robot="test_robot",
            reference="smpl",
            calibrated_joint_q={
                "left_shoulder_roll_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
            },
        ),
        target,
    )

    status = service.status(
        CalibrationStatusRequest(
            robot_id="test_robot",
            robot_asset_id=robot_asset_id,
            reference="smpl",
        )
    )

    assert status.state.value == "invalid"
    assert status.current_validation is not None
    assert status.current_validation.valid is False
    alignment = next(
        check
        for check in status.current_validation.checks
        if check.code == "CALIBRATION_POSE_ALIGNED"
    )
    assert alignment.level.value == "error"

    proposal = service.propose(
        CalibrationProposalRequest(
            robot_id="test_robot",
            robot_asset_id=robot_asset_id,
            reference="smpl",
        )
    )
    receipt = service.save(
        CalibrationSaveRequest(
            candidate_id=proposal.candidate.candidate_id,
            save_mode="validated_silent",
        )
    )
    assert receipt.previous_calibration_id == status.calibration_id
    assert receipt.previous_calibration_archived is True
    assert status.calibration_digest is not None
    assert (
        tmp_path / "agent-state" / "calibration-history" / f"{status.calibration_digest}.yaml"
    ).is_file()


def test_silent_save_rejects_a_candidate_when_the_baseline_changed(tmp_path: Path) -> None:
    service, robot_asset_id, user_root = _service(tmp_path)
    proposal = service.propose(
        CalibrationProposalRequest(
            robot_id="test_robot",
            robot_asset_id=robot_asset_id,
            reference="smpl",
        )
    )
    save_calibration(
        RobotRetargetCalibration(
            robot="test_robot",
            reference="smpl",
            calibrated_joint_q={"left_shoulder_roll_joint": 1.0},
        ),
        user_root / "test_robot" / "retarget_calibration_smpl.yaml",
    )

    with pytest.raises(CalibrationServiceError) as raised:
        service.save(
            CalibrationSaveRequest(
                candidate_id=proposal.candidate.candidate_id,
                save_mode="validated_silent",
            )
        )

    assert raised.value.code == "CALIBRATION_CANDIDATE_STALE"
