"""Application service for registering and inspecting Agent workflow assets.

``AssetRegistry`` owns filesystem authorization, immutable manifests, and
persistence. ``MotionAssetInspector`` owns format discovery and read-only
content checks. This module composes those two boundaries so transport adapters
do not need to exchange host paths or reimplement bundle assembly.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetCategory,
    AssetDetected,
    AssetFileRole,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    AssetSearchResponse,
    ErrorStage,
)

from .asset_inspection import (
    MotionAssetDiscoveryError,
    MotionAssetInspector,
    discover_primary,
)
from .assets import (
    AssetRegistry,
    AssetServiceError,
    DiscoveredAsset,
    DiscoveredAssetFile,
)
from .r2r_asset_inspection import (
    R2RTrajectoryDiscoveryError,
    R2RTrajectoryInspector,
    discover_r2r_trajectory,
)
from .robot_asset_inspection import (
    RobotAssetDiscoveryError,
    RobotAssetInspector,
    discover_robot_bundle,
)

_MOTION_EXTENSIONS = frozenset(
    {".bvh", ".csv", ".glb", ".gltf", ".npy", ".npz", ".pickle", ".pkl", ".pt", ".pth"}
)


def _safe_discovery_candidates(values: tuple[str, ...]) -> list[str]:
    """Keep only normalized relative candidate names in public error details."""

    safe: list[str] = []
    for value in values:
        if not value or "\\" in value:
            continue
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            continue
        safe.append(posix.as_posix())
    return safe


def _discovery_error(error: MotionAssetDiscoveryError) -> AssetServiceError:
    """Translate an inspector discovery failure to the shared service error."""

    details: dict[str, object] = {}
    candidates = _safe_discovery_candidates(error.candidates)
    if candidates:
        details["candidates"] = candidates
    return AssetServiceError(
        ApiError(
            code=error.code,
            message=str(error),
            retryable=False,
            stage=ErrorStage.ASSET_REGISTRATION,
            details=details,
        )
    )


def _robot_discovery_error(error: RobotAssetDiscoveryError) -> AssetServiceError:
    """Move safe robot-discovery diagnostics to the registration stage."""

    return AssetServiceError(
        error.api_error.model_copy(update={"stage": ErrorStage.ASSET_REGISTRATION})
    )


def _registration_kind(
    request: AssetRegistrationRequest,
    candidate: Path,
) -> AssetKind:
    """Resolve an explicit or unambiguous motion/robot registration kind."""

    if request.kind is not None:
        if request.kind not in {
            AssetKind.MOTION_BUNDLE,
            AssetKind.ROBOT_BUNDLE,
            AssetKind.ROBOT_TRAJECTORY_BUNDLE,
        }:
            raise AssetServiceError(
                ApiError(
                    code="UNSUPPORTED_ASSET_KIND",
                    message="Asset registration currently supports motion and robot bundles.",
                    stage=ErrorStage.ASSET_REGISTRATION,
                    details={"kind": request.kind.value},
                )
            )
        return request.kind
    if request.category is AssetCategory.ROBOT_TRAJECTORY:
        return AssetKind.ROBOT_TRAJECTORY_BUNDLE
    if request.category is AssetCategory.ROBOT_MODEL or candidate.suffix.casefold() == ".urdf":
        return AssetKind.ROBOT_BUNDLE
    if candidate.is_file():
        return AssetKind.MOTION_BUNDLE

    has_urdf = any(path.is_file() for path in candidate.rglob("*.urdf"))
    has_motion = any(
        path.is_file() and path.suffix.casefold() in _MOTION_EXTENSIONS
        for path in candidate.rglob("*")
    )
    if has_urdf and has_motion:
        raise AssetServiceError(
            ApiError(
                code="BUNDLE_AMBIGUOUS",
                message="The directory contains both robot and motion assets; specify kind.",
                stage=ErrorStage.ASSET_REGISTRATION,
            )
        )
    return AssetKind.ROBOT_BUNDLE if has_urdf else AssetKind.MOTION_BUNDLE


class AgentAssetService:
    """Transport-neutral facade for the Agent asset lifecycle."""

    def __init__(
        self,
        registry: AssetRegistry,
        inspector: MotionAssetInspector | None = None,
        robot_inspector: RobotAssetInspector | None = None,
        r2r_inspector: R2RTrajectoryInspector | None = None,
    ) -> None:
        self._registry = registry
        self._inspector = inspector or MotionAssetInspector()
        self._robot_inspector = robot_inspector or RobotAssetInspector()
        self._r2r_inspector = r2r_inspector or R2RTrajectoryInspector()

    @property
    def allowed_root_ids(self) -> tuple[str, ...]:
        """Return configured root identifiers without revealing their paths."""

        return self._registry.allowed_root_ids

    def registration_hint(
        self,
        trusted_path: Path,
        *,
        kind: AssetKind | None = None,
        category: AssetCategory | None = None,
        recursive: bool = True,
    ) -> AssetRegistrationRequest:
        """Return a portable request for a path chosen by trusted service code.

        Transport adapters must never expose ``trusted_path`` as an Agent
        parameter.  This method exists for composition code such as preflight,
        where an installed preset already owns the local path and an Agent
        needs an executable ``register_asset_bundle`` continuation.
        """

        return self._registry.registration_hint(
            trusted_path,
            kind=kind,
            category=category,
            recursive=recursive,
        )

    def register(self, request: AssetRegistrationRequest) -> AssetBundle:
        """Discover and persist one motion bundle below an allowlisted root."""

        candidate = self._registry.resolve_registration_path(request)
        kind = _registration_kind(request, candidate)
        r2r_discovery = None
        explicitly_r2r = (
            kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE
            or request.category is AssetCategory.ROBOT_TRAJECTORY
        )
        if explicitly_r2r and (
            request.category not in {None, AssetCategory.ROBOT_TRAJECTORY}
            or request.kind not in {None, AssetKind.ROBOT_TRAJECTORY_BUNDLE}
        ):
            raise AssetServiceError(
                ApiError(
                    code="ASSET_KIND_MISMATCH",
                    message="Robot trajectory bundles require the robot_trajectory category.",
                    stage=ErrorStage.ASSET_REGISTRATION,
                )
            )
        if explicitly_r2r or (request.kind is None and request.category is None):
            try:
                r2r_discovery = discover_r2r_trajectory(
                    candidate,
                    allow_code_capable=explicitly_r2r,
                )
            except R2RTrajectoryDiscoveryError as error:
                if explicitly_r2r:
                    raise _discovery_error(error) from error
            else:
                kind = AssetKind.ROBOT_TRAJECTORY_BUNDLE
        if kind is AssetKind.ROBOT_BUNDLE:
            try:
                robot = discover_robot_bundle(candidate)
            except RobotAssetDiscoveryError as error:
                raise _robot_discovery_error(error) from error
            discovery = DiscoveredAsset(
                primary_file=robot.primary_urdf,
                files=tuple(
                    DiscoveredAssetFile(
                        path=item.path,
                        role=item.role,
                        required=item.required,
                    )
                    for item in robot.files
                ),
                kind=AssetKind.ROBOT_BUNDLE,
                category=AssetCategory.ROBOT_MODEL,
                metadata=robot.metadata,
            )
        elif kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE:
            assert r2r_discovery is not None
            files = [
                DiscoveredAssetFile(
                    path=r2r_discovery.primary_path,
                    role=AssetFileRole.ROBOT_TRAJECTORY,
                    required=True,
                )
            ]
            for role, paths in sorted(
                r2r_discovery.sidecars.items(),
                key=lambda item: item[0].value,
            ):
                files.extend(
                    DiscoveredAssetFile(path=path, role=role, required=True) for path in paths
                )
            discovery = DiscoveredAsset(
                primary_file=r2r_discovery.primary_path,
                files=tuple(files),
                kind=AssetKind.ROBOT_TRAJECTORY_BUNDLE,
                category=AssetCategory.ROBOT_TRAJECTORY,
                detected=AssetDetected(
                    dataset="robot_trajectory",
                    recommended_backend="newton",
                    source_robot_id=r2r_discovery.source_robot_id,
                    trajectory_profile=r2r_discovery.profile,
                ),
            )
        else:
            try:
                discovered = discover_primary(candidate)
            except MotionAssetDiscoveryError as error:
                raise _discovery_error(error) from error

            files = [
                DiscoveredAssetFile(
                    path=discovered.primary_path,
                    role=AssetFileRole.MOTION,
                    required=True,
                )
            ]
            for role, paths in sorted(
                discovered.sidecars.items(),
                key=lambda item: item[0].value,
            ):
                files.extend(
                    DiscoveredAssetFile(path=path, role=role, required=True)
                    for path in paths
                    if path != discovered.primary_path
                )

            discovery = DiscoveredAsset(
                primary_file=discovered.primary_path,
                files=tuple(files),
                kind=AssetKind.MOTION_BUNDLE,
                category=discovered.category,
                detected=AssetDetected(
                    dataset=discovered.dataset,
                    reference=discovered.reference,
                    recommended_backend=discovered.recommended_backend,
                ),
            )
        return self._registry.register(request, discovery=discovery)

    def get(self, asset_id: str) -> AssetBundle:
        """Return one registered, portable asset manifest."""

        return self._registry.get(asset_id)

    def resolve_primary(self, asset_id: str, *, verify_hash: bool = True) -> Path:
        """Resolve the trusted primary file for an in-process executor.

        This is deliberately a Python-only application-service boundary.  REST,
        CLI, and MCP adapters must return the portable :class:`AssetBundle`
        instead of serializing this host path.
        """

        bundle = self._registry.get(asset_id)
        return self.resolve_file(
            bundle.asset_id,
            bundle.primary_file,
            verify_hash=verify_hash,
        )

    def resolve_file(
        self,
        asset_id: str,
        relative_path: str,
        *,
        verify_hash: bool = True,
    ) -> Path:
        """Resolve one manifest-declared file for trusted in-process consumers.

        The registry enforces membership, containment, and optional content-hash
        verification.  Protocol adapters must keep returning portable manifest
        paths instead of exposing this application-internal absolute path.
        """

        return self._registry.resolve_file(
            asset_id,
            relative_path,
            verify_hash=verify_hash,
        )

    def search(
        self,
        *,
        query: str | None = None,
        kind: AssetKind | str | None = None,
        category: AssetCategory | str | None = None,
        dataset: str | None = None,
        reference: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AssetSearchResponse:
        """Search registered manifests using the registry's bounded filters."""

        return self._registry.search(
            query=query,
            kind=kind,
            category=category,
            dataset=dataset,
            reference=reference,
            limit=limit,
            offset=offset,
        )

    def inspect(self, request: AssetInspectionRequest) -> AssetInspection:
        """Inspect a registered bundle after re-resolving its trusted base path."""

        bundle = self._registry.get(request.asset_id)
        primary = self._registry.resolve_file(
            bundle.asset_id,
            bundle.primary_file,
            verify_hash=False,
        )

        # The registry returns the resolved primary file, while the inspector
        # consumes the bundle base. Walk one parent per portable manifest path
        # component so nested primaries remain relative to the registered root.
        bundle_root = primary
        for _ in PurePosixPath(bundle.primary_file).parts:
            bundle_root = bundle_root.parent

        if bundle.kind is AssetKind.ROBOT_BUNDLE:
            inspector = self._robot_inspector
        elif bundle.kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE:
            inspector = self._r2r_inspector
        else:
            inspector = self._inspector
        return inspector.inspect(
            bundle,
            bundle_root,
            verify_hashes=request.verify_hashes,
            parse_content=request.parse_content,
        )


__all__ = ["AgentAssetService"]
