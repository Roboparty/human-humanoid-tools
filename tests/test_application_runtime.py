from __future__ import annotations

from hhtools.application.runtime import ApplicationPaths, build_application_runtime


def test_isolated_application_paths_keep_every_persistent_root_explicit(tmp_path) -> None:
    paths = ApplicationPaths.isolated(tmp_path)
    runtime = build_application_runtime(paths, max_retained_jobs=4)
    try:
        assert runtime.paths == paths
        assert runtime.state.source_root == paths.source_root
        assert runtime.state.save_dir == paths.save_dir
        assert runtime.state.robot_root == paths.robot_root
        assert runtime.state.job_history.root == paths.job_history_dir.resolve()
        assert runtime.job_settings_store is not None
        assert runtime.job_settings_store.path == paths.job_settings_path.resolve()
        assert runtime.motion_library_settings_store.path == (
            paths.motion_library_settings_path.resolve()
        )
        assert paths.motion_library_root.is_dir()
        assert runtime.services.agent_job_manager.execution_available is True
    finally:
        runtime.scheduler.shutdown(wait=True)
        runtime.lifecycle._cleanup_once()
        runtime.lifecycle.agent_lease.release()
