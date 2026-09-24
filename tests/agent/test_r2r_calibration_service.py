from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from hhtools.contracts import (
    AssetRegistrationRequest,
    R2RCalibrationPreviewRequest,
    R2RCalibrationProposalRequest,
    R2RCalibrationSaveRequest,
    R2RCalibrationStatusRequest,
    R2RCalibrationValidationRequest,
)
from hhtools.retarget.robot_to_robot import save_r2r_calibration
from hhtools.robot.loader import load_robot
from hhtools.robot.registry import list_presets_in_root_readonly, preset_from_dir
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.calibration import CalibrationServiceError
from hhtools.services.calibration_candidates import CalibrationCandidateStore
from hhtools.services.r2r_calibration import R2RCalibrationService


def _write_robot_bundle(model, root: Path) -> Path:
    target = root / model.preset.name
    target.mkdir(parents=True)
    shutil.copy2(model.preset.urdf_path, target / "robot.urdf")
    (target / "robot.yaml").write_text(
        yaml.safe_dump(
            {
                "name": model.preset.name,
                "display_name": model.preset.display_name,
                "urdf": "robot.urdf",
                "dof_order": list(model.dof_names()),
                "ik_map": dict(model.preset.ik_map),
                "feet": dict(model.preset.feet),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return target


def _service(
    tmp_path: Path,
    grounding_robot_pair,
    *,
    bundled_calibration: bool = False,
    shared_robot_root: bool = False,
):
    source_model, target_model = grounding_robot_pair
    robot_root = tmp_path / "robots"
    source_path = _write_robot_bundle(source_model, robot_root)
    target_path = _write_robot_bundle(target_model, robot_root)
    user_root = robot_root if shared_robot_root else tmp_path / "user-robots"
    user_root.mkdir(exist_ok=True)
    if bundled_calibration:
        save_r2r_calibration(
            target_path,
            target_robot=target_model.preset.name,
            source_robot=source_model.preset.name,
            calibrated_joint_q={name: 0.0 for name in target_model.dof_names()},
            user_root=user_root,
        )
    state_root = tmp_path / "agent-state"
    assets = AgentAssetService(AssetRegistry(state_root, {"robots": robot_root}))
    source_bundle = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path=source_path.name,
            kind="robot_bundle",
        )
    )
    target_bundle = assets.register(
        AssetRegistrationRequest(
            root_id="robots",
            relative_path=target_path.name,
            kind="robot_bundle",
        )
    )

    def materialize(robot_id: str, _asset_id: str):
        return load_robot(preset_from_dir(robot_root / robot_id), compile_mjcf=False)

    service = R2RCalibrationService(
        assets,
        CalibrationCandidateStore(state_root),
        robot_provider=lambda: list_presets_in_root_readonly(robot_root),
        materialize_robot=materialize,
        release_robot=lambda _model: None,
        user_robot_root=user_root,
        request_id_provider=lambda: "req_r2r_calibration_test",
    )
    identity = R2RCalibrationStatusRequest(
        source_robot_id=source_model.preset.name,
        source_robot_asset_id=source_bundle.asset_id,
        target_robot_id=target_model.preset.name,
        target_robot_asset_id=target_bundle.asset_id,
    )
    return service, identity, source_path, target_path, user_root


@pytest.mark.parametrize("replace_existing", [False, True])
def test_library_r2r_save_keeps_both_assets_and_survives_restart(
    tmp_path: Path,
    grounding_robot_pair,
    replace_existing: bool,
) -> None:
    service, identity, source, target, user_root = _service(
        tmp_path,
        grounding_robot_pair,
        shared_robot_root=True,
        bundled_calibration=replace_existing,
    )
    before = {p: p.read_bytes() for root in (source, target) for p in root.iterdir() if p.is_file()}
    proposal = service.propose(R2RCalibrationProposalRequest(**identity.model_dump()))
    request = R2RCalibrationSaveRequest(
        candidate_id=proposal.candidate.candidate_id,
        save_mode="validated_silent",
    )
    receipt = service.save(request)
    restarted = R2RCalibrationService(
        AgentAssetService(AssetRegistry(tmp_path / "agent-state", {"robots": user_root})),
        CalibrationCandidateStore(tmp_path / "agent-state"),
        robot_provider=service._robot_provider,
        materialize_robot=service._materialize_robot,
        release_robot=lambda _model: None,
        user_robot_root=user_root,
    )
    assert restarted.save(request) == receipt
    status = restarted.status(identity)
    assert status.state.value == "valid"
    assert status.source_robot_asset_id == identity.source_robot_asset_id
    assert status.target_robot_asset_id == identity.target_robot_asset_id
    assert status.calibration_digest == receipt.calibration_digest
    assert receipt.previous_calibration_archived is replace_existing
    assert {
        p: p.read_bytes() for root in (source, target) for p in root.iterdir() if p.is_file()
    } == before
    for root, asset_id in (
        (source, identity.source_robot_asset_id),
        (target, identity.target_robot_asset_id),
    ):
        assert (
            restarted._asset_service.register(
                AssetRegistrationRequest(
                    root_id="robots",
                    relative_path=root.name,
                    kind="robot_bundle",
                )
            ).asset_id
            == asset_id
        )


def test_r2r_proposal_preview_and_silent_save_round_trip(
    tmp_path: Path,
    grounding_robot_pair,
) -> None:
    service, identity, _source_path, target_path, user_root = _service(
        tmp_path,
        grounding_robot_pair,
    )

    before = service.status(identity)
    proposal = service.propose(R2RCalibrationProposalRequest(**identity.model_dump()))
    candidate_id = proposal.candidate.candidate_id
    validation = service.validate(R2RCalibrationValidationRequest(candidate_id=candidate_id))
    preview, image = service.preview(R2RCalibrationPreviewRequest(candidate_id=candidate_id))
    receipt = service.save(
        R2RCalibrationSaveRequest(
            candidate_id=candidate_id,
            save_mode="gpt_vision_silent",
            visual_review={
                "reviewer": "gpt_vision",
                "verdict": "pass",
                "model_hint": "gpt-r2r-test",
                "summary": "The source reference and target pose align in both views.",
            },
        )
    )
    replayed = service.save(
        R2RCalibrationSaveRequest(
            candidate_id=candidate_id,
            save_mode="gpt_vision_silent",
            visual_review={
                "reviewer": "gpt_vision",
                "verdict": "pass",
                "model_hint": "gpt-r2r-test",
                "summary": "The source reference and target pose align in both views.",
            },
        )
    )
    after = service.status(identity)

    assert before.state.value == "missing"
    assert proposal.candidate.workflow == "r2r"
    assert proposal.validation.valid is True
    assert validation.valid is True
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert preview.sha256 == hashlib.sha256(image).hexdigest()
    assert receipt == replayed
    assert after.state.value == "valid"
    assert after.calibration_id == receipt.calibration_id
    assert after.storage == "user_calibration"
    saved = (
        user_root / identity.target_robot_id / f"r2r_calibration_{identity.source_robot_id}.yaml"
    )
    assert saved.is_file()
    assert not (target_path / saved.name).exists()
    saved_document = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert candidate_id in saved_document["notes"]
    assert "model=gpt-r2r-test" in saved_document["notes"]


def test_r2r_replacement_archives_the_previous_pair_calibration(
    tmp_path: Path,
    grounding_robot_pair,
) -> None:
    service, identity, _source_path, target_path, user_root = _service(
        tmp_path,
        grounding_robot_pair,
    )
    initial = save_r2r_calibration(
        target_path,
        target_robot=identity.target_robot_id,
        source_robot=identity.source_robot_id,
        calibrated_joint_q={name: 0.0 for name in grounding_robot_pair[1].dof_names()},
        user_root=user_root,
        prefer_user_overlay=True,
    )
    previous_payload = initial.read_bytes()
    previous_digest = hashlib.sha256(previous_payload).hexdigest()
    spine_joint = next(
        name for name in grounding_robot_pair[1].dof_names() if name == "spine_joint"
    )
    proposal = service.propose(
        R2RCalibrationProposalRequest(
            **identity.model_dump(),
            joint_q_overrides={spine_joint: 0.1},
            locked_joints=[spine_joint],
        )
    )

    receipt = service.save(
        R2RCalibrationSaveRequest(
            candidate_id=proposal.candidate.candidate_id,
            save_mode="validated_silent",
        )
    )

    assert proposal.validation.valid is True
    assert receipt.previous_calibration_id == f"cal:sha256:{previous_digest}"
    assert receipt.previous_calibration_archived is True
    assert (
        tmp_path / "agent-state" / "calibration-history" / f"{previous_digest}.yaml"
    ).read_bytes() == previous_payload


def test_silent_save_adopts_an_identical_bundled_pose_into_user_overlay(
    tmp_path: Path,
    grounding_robot_pair,
) -> None:
    service, identity, _source_path, target_path, user_root = _service(
        tmp_path,
        grounding_robot_pair,
        bundled_calibration=True,
    )
    bundled = target_path / f"r2r_calibration_{identity.source_robot_id}.yaml"
    bundled_payload = bundled.read_bytes()
    before = service.status(identity)
    proposal = service.propose(R2RCalibrationProposalRequest(**identity.model_dump()))

    receipt = service.save(
        R2RCalibrationSaveRequest(
            candidate_id=proposal.candidate.candidate_id,
            save_mode="validated_silent",
        )
    )
    after = service.status(identity)

    override = user_root / identity.target_robot_id / bundled.name
    assert before.storage == "robot_bundle"
    assert after.storage == "user_calibration"
    assert override.is_file()
    assert bundled.read_bytes() == bundled_payload
    assert receipt.previous_calibration_archived is True


def test_r2r_save_rejects_a_pair_baseline_created_after_proposal(
    tmp_path: Path,
    grounding_robot_pair,
) -> None:
    service, identity, _source_path, target_path, user_root = _service(
        tmp_path,
        grounding_robot_pair,
    )
    proposal = service.propose(R2RCalibrationProposalRequest(**identity.model_dump()))
    external = save_r2r_calibration(
        target_path,
        target_robot=identity.target_robot_id,
        source_robot=identity.source_robot_id,
        calibrated_joint_q={name: 0.0 for name in grounding_robot_pair[1].dof_names()},
        user_root=user_root,
        prefer_user_overlay=True,
        notes="external calibration",
    )
    external_payload = external.read_bytes()

    with pytest.raises(CalibrationServiceError) as raised:
        service.save(
            R2RCalibrationSaveRequest(
                candidate_id=proposal.candidate.candidate_id,
                save_mode="validated_silent",
            )
        )

    assert raised.value.code == "CALIBRATION_CANDIDATE_STALE"
    assert external.read_bytes() == external_payload
