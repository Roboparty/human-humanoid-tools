from pathlib import Path

from hhtools.web.jobs.batch_failure_log import open_batch_failure_log


def test_local_batch_failure_keeps_source_in_place(tmp_path: Path) -> None:
    source = tmp_path / "motions" / "walk.npz"
    source.parent.mkdir()
    source.write_bytes(b"motion")
    failure_log = open_batch_failure_log(tmp_path / "save", "job-1", "batch")

    item = failure_log.record(
        {
            "origin": "local",
            "source_path": str(source),
            "stem": "walk",
        },
        stage="load",
        reason="invalid clip",
    )
    failure_log.finalize(job_id="job-1", out_name="batch")

    assert item["log_rel"] is None
    assert source.read_bytes() == b"motion"
    assert {path.name for path in failure_log.root.iterdir()} == {
        "failures.json",
        "失败说明.txt",
    }
    assert str(source) in (failure_log.root / "失败说明.txt").read_text()
