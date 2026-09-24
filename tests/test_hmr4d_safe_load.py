from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from hhtools.io.datasets.hmr4d import _load_hmr4d
from hhtools.io.mimic_detect import sniff_pt_dataset


def _write_marker(path: str) -> None:
    Path(path).write_text("unsafe pickle executed", encoding="utf-8")


class _UnsafePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_hmr4d_loader_accepts_the_expected_tensor_dictionary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "hmr4d_results.pt"
    torch.save(
        {
            "smpl_params_global": {
                "body_pose": torch.zeros((2, 63)),
                "betas": torch.zeros((2, 10)),
                "global_orient": torch.zeros((2, 3)),
                "transl": torch.zeros((2, 3)),
            }
        },
        path,
    )
    monkeypatch.setattr("hhtools.bodymodels.paths.find_body_model", lambda *_args: None)

    params = _load_hmr4d(path)

    assert params.num_frames == 2
    assert params.body_pose.shape == (2, 63)
    assert params.trans.shape == (2, 3)


def test_hmr4d_loader_does_not_execute_pickle_globals(tmp_path: Path) -> None:
    path = tmp_path / "malicious.pt"
    marker = tmp_path / "executed.txt"
    torch.save({"smpl_params_global": _UnsafePayload(marker)}, path)

    with pytest.raises(pickle.UnpicklingError):
        _load_hmr4d(path)

    assert not marker.exists()


def test_hmr4d_sniffer_does_not_execute_pickle_globals(tmp_path: Path) -> None:
    path = tmp_path / "malicious.pt"
    marker = tmp_path / "sniffer-executed.txt"
    torch.save({"smpl_params_global": _UnsafePayload(marker)}, path)

    # Unknown PT files retain the existing GVHMR fallback classification, but
    # classification must never execute pickle globals merely to inspect them.
    assert sniff_pt_dataset(path) == "gvhmr"
    assert not marker.exists()
