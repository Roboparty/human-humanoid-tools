"""Asset bundle and inspection contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from .common import ApiError, AssetId, ContractModel, SchemaVersion, Sha256Hex


class AssetKind(StrEnum):
    """Logical type of a registered asset."""

    MOTION_BUNDLE = "motion_bundle"
    ROBOT_TRAJECTORY_BUNDLE = "robot_trajectory_bundle"
    ROBOT_BUNDLE = "robot_bundle"
    CALIBRATION_BUNDLE = "calibration_bundle"
    DATASET_BUNDLE = "dataset_bundle"
    VIDEO = "video"


class AssetCategory(StrEnum):
    """Workflow category used for backend selection."""

    PLAIN_MOTION = "plain_motion"
    OBJECT_INTERACTION = "object_interaction"
    TERRAIN_SCENE = "terrain_scene"
    ROBOT_TRAJECTORY = "robot_trajectory"
    ROBOT_MODEL = "robot_model"
    CALIBRATION = "calibration"


class AssetFileRole(StrEnum):
    """Semantic role of a file inside a bundle."""

    MOTION = "motion"
    ROBOT_TRAJECTORY = "robot_trajectory"
    ROBOT_DESCRIPTION = "robot_description"
    VISUAL_MESH = "visual_mesh"
    COLLISION_MESH = "collision_mesh"
    OBJECT_MESH = "object_mesh"
    TERRAIN_MESH = "terrain_mesh"
    OBJECT_TRAJECTORY = "object_trajectory"
    CALIBRATION = "calibration"
    METADATA = "metadata"
    VIDEO = "video"
    OTHER = "other"


class InspectionStatus(StrEnum):
    """Machine-readable outcome of inspecting an asset."""

    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


class AssetSourceScheme(StrEnum):
    """Controlled source schemes understood by the AssetRegistry."""

    MANAGED_FILE = "managed_file"
    UPLOAD = "upload"
    SHARED_STORAGE = "shared_storage"
    ARTIFACT = "artifact"


class AssetSource(ContractModel):
    """Location identity without exposing an arbitrary host absolute path."""

    scheme: AssetSourceScheme
    root_id: Annotated[str, Field(min_length=1, max_length=128)]
    registered_at: AwareDatetime
    logical_path: Annotated[str | None, Field(default=None, min_length=1, max_length=1024)]

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str | None) -> str | None:
        return None if value is None else _validate_bundle_path(value)


class AssetDetected(ContractModel):
    """Small set of routing hints discovered from a registered bundle."""

    dataset: str | None = None
    reference: str | None = None
    recommended_backend: str | None = None
    source_robot_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    trajectory_profile: Literal["mimic", "intermimic", "meshmimic"] | None = None


# Backwards-compatible import name for the earliest service prototype.
DetectedAssetMetadata = AssetDetected


def _validate_bundle_path(value: str) -> str:
    """Require a portable, bundle-relative path without traversal."""

    if not value:
        raise ValueError("bundle path must not be empty")
    if "\\" in value:
        raise ValueError("bundle paths must use forward slashes")

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("bundle path must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("bundle path must be normalized and cannot traverse parents")
    return posix_path.as_posix()


class AssetFile(ContractModel):
    """One content-addressed file inside an :class:`AssetBundle`."""

    role: AssetFileRole
    relative_path: Annotated[
        str,
        Field(min_length=1, max_length=1024, description="Portable path relative to the bundle."),
    ]
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    media_type: str | None = Field(default=None, description="IANA media type when known.")
    required: bool = True

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_bundle_path(value)


class AssetBundle(ContractModel):
    """Portable manifest for all files required by one logical input."""

    schema_version: SchemaVersion = SchemaVersion.V1
    asset_id: AssetId
    kind: AssetKind
    category: AssetCategory
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    primary_file: Annotated[str, Field(min_length=1, max_length=1024)]
    files: Annotated[list[AssetFile], Field(min_length=1)]
    source: AssetSource | None = None
    detected: AssetDetected | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("primary_file")
    @classmethod
    def validate_primary_file(cls, value: str) -> str:
        return _validate_bundle_path(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> AssetBundle:
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("asset bundle contains duplicate relative paths")
        if self.primary_file not in paths:
            raise ValueError("primary_file must reference a file in the bundle")
        return self


class AssetInspection(ContractModel):
    """Compact, structured facts discovered without running retargeting."""

    schema_version: SchemaVersion = SchemaVersion.V1
    asset_id: AssetId
    status: InspectionStatus
    kind: AssetKind
    category: AssetCategory
    source_format: str | None = Field(default=None, description="Detected source format.")
    dataset: str | None = None
    reference_model: str | None = Field(
        default=None,
        description="Detected human reference, for example smpl, smplh, or smplx.",
    )
    frame_count: Annotated[int | None, Field(default=None, ge=0)]
    frame_rate_hz: Annotated[float | None, Field(default=None, gt=0)]
    duration_seconds: Annotated[float | None, Field(default=None, ge=0)]
    joint_count: Annotated[int | None, Field(default=None, ge=0)]
    source_robot_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    has_object: bool = False
    has_terrain: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[ApiError] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> AssetInspection:
        if self.status is InspectionStatus.INVALID and not self.errors:
            raise ValueError("invalid inspections must include at least one error")
        if self.errors and self.status is not InspectionStatus.INVALID:
            raise ValueError("inspections with errors must use invalid status")
        if self.status is InspectionStatus.VALID and (self.warnings or self.errors):
            raise ValueError("valid inspections cannot include warnings or errors")
        if self.status is InspectionStatus.VALID_WITH_WARNINGS and not self.warnings:
            raise ValueError("valid_with_warnings inspections must include a warning")
        return self


class AssetRegistrationRequest(ContractModel):
    """Register a bundle from a path below a server-configured root."""

    schema_version: SchemaVersion = SchemaVersion.V1
    root_id: Annotated[str, Field(min_length=1, max_length=128)]
    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    display_name: Annotated[str | None, Field(default=None, max_length=256)]
    kind: AssetKind | None = None
    category: AssetCategory | None = None
    recursive: bool = True

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_bundle_path(value)


class AssetInspectionRequest(ContractModel):
    """Request integrity and parse checks for an existing registered asset."""

    schema_version: SchemaVersion = SchemaVersion.V1
    asset_id: AssetId
    verify_hashes: bool = True
    parse_content: bool = True


class AssetSearchResponse(ContractModel):
    """Versioned, bounded search result for registered asset manifests."""

    schema_version: SchemaVersion = SchemaVersion.V1
    assets: list[AssetBundle] = Field(default_factory=list)
    total: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=500)]
    offset: Annotated[int, Field(ge=0)]


class AvailableAssetCatalogRequest(ContractModel):
    """Bounded filters for discovering registerable assets below configured roots."""

    schema_version: SchemaVersion = SchemaVersion.V1
    root_id: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        ),
    ]
    query: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    kind: AssetKind | None = None
    limit: Annotated[int, Field(ge=1, le=500)] = 100
    offset: Annotated[int, Field(ge=0)] = 0


class AvailableAssetCatalogEntry(ContractModel):
    """Portable registration metadata for one allowlisted, available asset."""

    root_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        ),
    ]
    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    kind: AssetKind
    category: AssetCategory
    recursive: bool = True
    dataset: Annotated[str | None, Field(default=None, min_length=1, max_length=128)]
    reference: Annotated[str | None, Field(default=None, min_length=1, max_length=128)]

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_bundle_path(value)


class AvailableAssetCatalogResponse(ContractModel):
    """Deterministic page of assets that can be registered without host paths."""

    schema_version: SchemaVersion = SchemaVersion.V1
    assets: list[AvailableAssetCatalogEntry] = Field(default_factory=list)
    total: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=500)]
    offset: Annotated[int, Field(ge=0)]
