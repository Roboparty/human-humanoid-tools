"""Regression guard for HHTools' Python package dependency direction.

The target graph keeps numerical/domain code independent from application
services and host adapters. ``web`` and ``mcp`` are sibling hosts: they may
share lower layers, but they must not import one another. ``viewer`` is now a
compatibility namespace and may only project migrated lower-layer helpers.

This test deliberately parses source with :mod:`ast` instead of importing the
package.  Besides keeping optional runtime dependencies out of architecture
tests, that also sees imports hidden inside functions and ``TYPE_CHECKING``
blocks.  A precise baseline freezes existing debt while every new violation
fails.  Removing debt must shrink the baseline in the same change so a later
commit cannot silently reintroduce it.
"""

from __future__ import annotations

import ast
import importlib.util
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "hhtools"
_MODULE_SCOPE = "<module>"
_UNRESOLVED_DYNAMIC_MODULE = "<unresolved-dynamic-import>"

# A source package may import only the listed direct dependencies (plus itself).
# Do not list every theoretically lower layer: an unused entry would silently
# pre-approve a new dependency without an architectural review.  When the last
# real import for an edge disappears, the policy test requires removing that
# edge here in the same change.
ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    "__root__": frozenset({"_version"}),
    "_version": frozenset(),
    "utils": frozenset(),
    "contracts": frozenset(),
    "core": frozenset(),
    "agent": frozenset({"contracts", "services"}),
    "application": frozenset(
        {
            "agent",
            "analysis",
            "contracts",
            "core",
            "io",
            "retarget",
            "robot",
            "services",
            "utils",
        }
    ),
    "bodymodels": frozenset({"core", "utils"}),
    "human": frozenset({"core"}),
    "robot": frozenset({"core", "utils"}),
    "io": frozenset({"bodymodels", "core", "robot"}),
    "integrations": frozenset(),
    "retarget": frozenset({"bodymodels", "core", "io", "robot", "utils"}),
    "analysis": frozenset({"core", "io", "retarget", "robot", "services"}),
    "services": frozenset(
        {"_version", "contracts", "core", "io", "retarget", "robot", "utils"}
    ),
    "web": frozenset(
        {
            "analysis",
            "agent",
            "application",
            "contracts",
            "core",
            "integrations",
            "io",
            "retarget",
            "robot",
            "services",
            "utils",
        }
    ),
    "viewer": frozenset({"core", "services"}),
    "mcp": frozenset({"_version", "application", "contracts", "retarget", "services"}),
    "cli": frozenset({"_version", "bodymodels", "contracts", "core", "io", "retarget", "robot"}),
}

# These command modules are composition roots that start a host.  The exception
# is intentionally path-and-module-specific: even a launcher must not gain
# blanket access to every Web or Viewer implementation module.
CLI_HOST_LAUNCH_EXCEPTIONS: dict[str, frozenset[str]] = {
    "hhtools/cli/desktop_sidecar.py": frozenset({"hhtools.web.dependencies", "hhtools.web.server"}),
    "hhtools/cli/web.py": frozenset({"hhtools.web.dependencies", "hhtools.web.server"}),
}


@dataclass(frozen=True, order=True)
class ImportIdentity:
    """Stable identity for one kind of forbidden import.

    Line numbers are intentionally absent: moving otherwise unchanged code
    must not manufacture new debt.  The occurrence count still detects a
    second import in the same lexical scope.
    """

    source_path: str
    target_module: str
    scope: str
    fingerprint: str = ""


@dataclass(frozen=True)
class InternalImport:
    source_path: str
    source_module: str
    target_module: str
    scope: str
    line: int
    fingerprint: str

    @property
    def identity(self) -> ImportIdentity:
        return ImportIdentity(
            self.source_path,
            self.target_module,
            self.scope,
            self.fingerprint,
        )


# Current static debt: 26 forbidden import occurrences represented by 25 locations.
# Reduce this table whenever an inversion is removed.  Never add an entry merely
# to make a new failure green; fix the dependency direction instead.
_LEGACY_LOCATION_BASELINE: dict[ImportIdentity, int] = {
    ImportIdentity(
        "hhtools/core/grounding.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "preferred_floor_contact_bone_indices",
    ): 1,
    ImportIdentity(
        "hhtools/io/bvh_detect.py",
        "hhtools.retarget.newton_basic.human_aliases",
        _MODULE_SCOPE,
    ): 1,
    ImportIdentity(
        "hhtools/io/datasets/meshmimic_holosoma.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "_load_terrain_heightfield",
    ): 1,
    ImportIdentity(
        "hhtools/io/datasets/parc_ms.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "ParcMsAdapter.load_motion",
    ): 2,
    ImportIdentity(
        "hhtools/io/mocap_parkour_import.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        _MODULE_SCOPE,
    ): 1,
    ImportIdentity(
        "hhtools/io/npz.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "_resolve_terrain_for_npz",
    ): 1,
    ImportIdentity(
        "hhtools/io/parc_export.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        _MODULE_SCOPE,
    ): 1,
    ImportIdentity(
        "hhtools/robot/ik_map_policy.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "is_omnicontact_like_motion",
    ): 1,
    ImportIdentity(
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "_newest_calibration_mtime",
    ): 1,
    ImportIdentity(
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "all_calibration_scales_for_preset",
    ): 1,
    ImportIdentity(
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "infer_joint_scales_from_urdf",
    ): 1,
    ImportIdentity(
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "joint_scale_baselines_for_preset",
    ): 1,
    ImportIdentity(
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "robot_dir_has_calibration",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration",
        "build_scaler_config_for_robot",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        _MODULE_SCOPE,
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "_shoulder_roll_scale_ratios",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "_yaml_active_scale_edits",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "default_human_height",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.config",
        _MODULE_SCOPE,
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.config",
        "build_feet_stabilizer_config",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.ground_collision_bodies",
        "_resolve_ground_collision_bodies",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "_apply_joint_scale_overrides_to_config",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.pipeline",
        "build_pipeline_config_for_preset",
    ): 1,
    ImportIdentity(
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.scaler",
        "resolve_retarget_scaler_config",
    ): 1,
    ImportIdentity(
        "hhtools/robot/standing_height.py",
        "hhtools.retarget.interaction_mesh.mujoco_scene",
        "estimate_robot_standing_height",
    ): 1,
}

# Imported symbols are a part of the baseline identity.  Without this second,
# static snapshot, an existing forbidden ``from`` import could silently grow
# from one private helper to several while retaining the same package edge.
_LEGACY_IMPORT_FINGERPRINTS: dict[tuple[str, str, str], str] = {
    (
        "hhtools/core/grounding.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "preferred_floor_contact_bone_indices",
    ): "from:auto_source_to_canonical",
    (
        "hhtools/io/bvh_detect.py",
        "hhtools.retarget.newton_basic.human_aliases",
        _MODULE_SCOPE,
    ): "from:is_mixamo_cmu_like,is_mocap_spine3_bvh_like,is_soma_bvh_like,is_xsens_mocap_like",
    (
        "hhtools/io/datasets/meshmimic_holosoma.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "_load_terrain_heightfield",
    ): "from:obj_to_heightfield",
    (
        "hhtools/io/datasets/parc_ms.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "ParcMsAdapter.load_motion",
    ): "from:obj_to_heightfield",
    (
        "hhtools/io/mocap_parkour_import.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        _MODULE_SCOPE,
    ): "from:posed_meshes_to_heightfield",
    (
        "hhtools/io/npz.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        "_resolve_terrain_for_npz",
    ): "from:obj_to_heightfield",
    (
        "hhtools/io/parc_export.py",
        "hhtools.retarget.interaction_mesh.heightfield",
        _MODULE_SCOPE,
    ): "from:obj_to_heightfield",
    (
        "hhtools/robot/ik_map_policy.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "is_omnicontact_like_motion",
    ): "from:is_two_segment_mixamo_like",
    (
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "_newest_calibration_mtime",
    ): "from:resolve_preset_calibration_file",
    (
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "all_calibration_scales_for_preset",
    ): "from:derive_calibration_params,load_calibration,resolve_preset_calibration_file",
    (
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "infer_joint_scales_from_urdf",
    ): "from:RobotRetargetCalibration,derive_calibration_params",
    (
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "joint_scale_baselines_for_preset",
    ): "from:derive_calibration_params,load_calibration,resolve_preset_calibration_file",
    (
        "hhtools/robot/joint_scales.py",
        "hhtools.retarget.calibration.calibration",
        "robot_dir_has_calibration",
    ): "from:resolve_calibration_file",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration",
        "build_scaler_config_for_robot",
    ): "from:build_scaler_config_from_calibration",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        _MODULE_SCOPE,
    ): "from:RobotRetargetCalibration",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "_shoulder_roll_scale_ratios",
    ): "from:load_calibration,resolve_preset_calibration_file",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "_yaml_active_scale_edits",
    ): "from:derive_calibration_params,normalize_calibration_reference",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.calibration.calibration",
        "default_human_height",
    ): "from:normalize_calibration_reference",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.config",
        _MODULE_SCOPE,
    ): "from:FeetStabilizerConfig,ScalerConfig,load_scaler_config",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.config",
        "build_feet_stabilizer_config",
    ): "from:ArmChainConfig",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.ground_collision_bodies",
        "_resolve_ground_collision_bodies",
    ): "from:build_ground_collision_bodies_from_ik_map",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.human_aliases",
        "_apply_joint_scale_overrides_to_config",
    ): "from:auto_source_to_canonical",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.pipeline",
        "build_pipeline_config_for_preset",
    ): "from:PipelineConfig",
    (
        "hhtools/robot/retarget_profile.py",
        "hhtools.retarget.newton_basic.scaler",
        "resolve_retarget_scaler_config",
    ): "from:adapt_scaler_config_for_hierarchy",
    (
        "hhtools/robot/standing_height.py",
        "hhtools.retarget.interaction_mesh.mujoco_scene",
        "estimate_robot_standing_height",
    ): "from:require_mujoco_model",
}

LEGACY_VIOLATION_BASELINE: dict[ImportIdentity, int] = {
    ImportIdentity(
        location.source_path,
        location.target_module,
        location.scope,
        _LEGACY_IMPORT_FINGERPRINTS[(location.source_path, location.target_module, location.scope)],
    ): count
    for location, count in _LEGACY_LOCATION_BASELINE.items()
}
LEGACY_VIOLATION_BASELINE.update(
    {
        ImportIdentity(
            "hhtools/cli/main.py",
            _UNRESOLVED_DYNAMIC_MODULE,
            "_attach",
            "dynamic:importlib.import_module:module_path",
        ): 1,
        ImportIdentity(
            "hhtools/retarget/__init__.py",
            _UNRESOLVED_DYNAMIC_MODULE,
            "__getattr__",
            "dynamic:importlib.import_module:f'{__name__}.{name}'",
        ): 1,
    }
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_REPOSITORY_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dependency_group(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "__root__"


class _ImportVisitor(ast.NodeVisitor):
    """Collect internal imports while retaining their lexical owner."""

    def __init__(self, *, source_path: str, source_module: str, is_package: bool) -> None:
        self.source_path = source_path
        self.source_module = source_module
        self.is_package = is_package
        self.scope: list[str] = []
        self.imports: list[InternalImport] = []
        self._importlib_names: set[str] = set()
        self._import_module_names = {"__import__"}

    @property
    def source_package(self) -> str:
        if self.is_package:
            return self.source_module
        return self.source_module.rpartition(".")[0]

    def _record(
        self,
        target: str | None,
        node: ast.Import | ast.ImportFrom | ast.Call,
        *,
        fingerprint: str,
    ) -> None:
        if target is None or (target != "hhtools" and not target.startswith("hhtools.")):
            return
        self.imports.append(
            InternalImport(
                source_path=self.source_path,
                source_module=self.source_module,
                target_module=target,
                scope=".".join(self.scope) or _MODULE_SCOPE,
                line=node.lineno,
                fingerprint=fingerprint,
            )
        )

    def _record_unresolved_dynamic(
        self,
        node: ast.Call,
        *,
        callee: str,
        name_expression: ast.expr | None,
    ) -> None:
        expression = "<missing>" if name_expression is None else ast.unparse(name_expression)
        self.imports.append(
            InternalImport(
                source_path=self.source_path,
                source_module=self.source_module,
                target_module=_UNRESOLVED_DYNAMIC_MODULE,
                scope=".".join(self.scope) or _MODULE_SCOPE,
                line=node.lineno,
                fingerprint=f"dynamic:{callee}:{expression}",
            )
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "importlib":
                self._importlib_names.add(alias.asname or "importlib")
            elif alias.name == "importlib.util" and alias.asname is None:
                # ``import importlib.util`` binds the top-level ``importlib``
                # name, from which ``import_module`` remains reachable.
                self._importlib_names.add("importlib")
            self._record(alias.name, node, fingerprint="import")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        target = self._resolve_import_from(node)
        if target == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    self._import_module_names.add(alias.asname or alias.name)

        # ``from hhtools import web`` references the child package, whereas
        # ordinary from-imports reference their declared module.
        if target == "hhtools":
            for alias in node.names:
                self._record(
                    target if alias.name == "*" else f"{target}.{alias.name}",
                    node,
                    fingerprint=f"from:{alias.name}",
                )
            return
        imported_symbols = ",".join(sorted(alias.name for alias in node.names))
        self._record(target, node, fingerprint=f"from:{imported_symbols}")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Catch both resolvable and computed dynamic imports."""

        callee: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id in self._import_module_names:
            callee = "__import__" if node.func.id == "__import__" else "importlib.import_module"
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._importlib_names
        ):
            callee = "importlib.import_module"

        if callee is not None:
            name_expression = self._call_argument(node, position=0, keyword="name")
            target = self._static_string(name_expression)
            if target is not None and target.startswith("."):
                package_expression = self._call_argument(node, position=1, keyword="package")
                package = self._dynamic_package(package_expression)
                if package is None:
                    target = None
                else:
                    try:
                        target = importlib.util.resolve_name(target, package)
                    except ImportError:
                        target = None

            if target is None:
                self._record_unresolved_dynamic(
                    node,
                    callee=callee,
                    name_expression=name_expression,
                )
            else:
                self._record(target, node, fingerprint=f"dynamic:{callee}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_scoped(node)

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _resolve_import_from(self, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        relative_name = "." * node.level + (node.module or "")
        try:
            return importlib.util.resolve_name(relative_name, self.source_package)
        except ImportError as error:
            raise AssertionError(
                f"Cannot resolve relative import {relative_name!r} in {self.source_path}:"
                f"{node.lineno}."
            ) from error

    @staticmethod
    def _call_argument(node: ast.Call, *, position: int, keyword: str) -> ast.expr | None:
        if len(node.args) > position:
            return node.args[position]
        return next((item.value for item in node.keywords if item.arg == keyword), None)

    @classmethod
    def _static_string(cls, node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = cls._static_string(node.left)
            right = cls._static_string(node.right)
            if left is not None and right is not None:
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    return None
                parts.append(value.value)
            return "".join(parts)
        return None

    def _dynamic_package(self, node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Name) and node.id == "__package__":
            return self.source_package
        return self._static_string(node)


def _scan_internal_imports() -> list[InternalImport]:
    imports: list[InternalImport] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        source_module = _module_name(path)
        source_path = path.relative_to(_REPOSITORY_ROOT).as_posix()
        visitor = _ImportVisitor(
            source_path=source_path,
            source_module=source_module,
            is_package=path.name == "__init__.py",
        )
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=source_path))
        imports.extend(visitor.imports)
    return imports


def _scan_source(
    source: str,
    *,
    source_module: str = "hhtools.core.architecture_fixture",
    is_package: bool = False,
) -> list[InternalImport]:
    """Parse an in-memory fixture with the same visitor used for the repository."""

    visitor = _ImportVisitor(
        source_path="hhtools/core/architecture_fixture.py",
        source_module=source_module,
        is_package=is_package,
    )
    visitor.visit(ast.parse(source))
    return visitor.imports


def _is_allowed(item: InternalImport) -> bool:
    if item.target_module == _UNRESOLVED_DYNAMIC_MODULE:
        return False
    source_group = _dependency_group(item.source_module)
    target_group = _dependency_group(item.target_module)
    if source_group == target_group:
        return True
    if target_group in ALLOWED_DEPENDENCIES.get(source_group, frozenset()):
        return True
    return item.target_module in CLI_HOST_LAUNCH_EXCEPTIONS.get(
        item.source_path,
        frozenset(),
    )


def _fix_hint(source_group: str, target_group: str) -> str:
    if target_group == "__root__":
        return "Replace or explicitly inventory the computed dynamic import."
    hosts = {"cli", "mcp", "viewer", "web"}
    if source_group in hosts and target_group in hosts:
        if source_group == "cli":
            return (
                "Only the two explicit CLI launch modules may start Web; "
                "move reusable behavior below the host adapters."
            )
        return "Sibling hosts must share contracts/services instead of importing each other."
    if target_group in hosts:
        return (
            "Lower packages must not depend on a UI/transport host; move shared logic to "
            "core, a domain package, or services."
        )
    if source_group == "core":
        return "Keep core foundational; move the primitive down or inject the policy from above."
    return (
        "Move the shared primitive to a lower package, or invert the dependency through a "
        "small contract/service."
    )


def _format_regressions(
    regressions: Counter[ImportIdentity],
    occurrences: dict[ImportIdentity, list[InternalImport]],
) -> str:
    lines = [
        "HHTools Python dependency direction regressed.",
        "New forbidden imports (source:line [scope] -> target):",
    ]
    for identity in sorted(regressions):
        found = sorted(occurrences[identity], key=lambda item: item.line)
        baseline_count = LEGACY_VIOLATION_BASELINE.get(identity, 0)
        locations = ", ".join(str(item.line) for item in found)
        item = found[0]
        source_group = _dependency_group(item.source_module)
        target_group = _dependency_group(item.target_module)
        lines.extend(
            (
                f"- {identity.source_path}:{locations} [{identity.scope}]",
                f"  {item.source_module} -> {identity.target_module} "
                f"({source_group} -> {target_group}); baseline allows "
                f"{baseline_count}, found {len(found)}.",
                f"  Import fingerprint: {identity.fingerprint}",
                f"  Fix: {_fix_hint(source_group, target_group)}",
            )
        )
    lines.append(
        "Do not extend LEGACY_VIOLATION_BASELINE for new code; it exists only to burn down "
        "the 2026-09-02 legacy snapshot."
    )
    return "\n".join(lines)


def _format_removed_debt(removed: Counter[ImportIdentity]) -> str:
    lines = ["Architecture debt was removed; shrink LEGACY_VIOLATION_BASELINE:"]
    for identity in sorted(removed):
        lines.append(
            f"- {identity.source_path} [{identity.scope}] -> {identity.target_module} "
            f"[{identity.fingerprint}] (remove {removed[identity]} occurrence(s))"
        )
    return "\n".join(lines)


def test_dependency_policy_covers_packages_without_preapproving_edges() -> None:
    imports = _scan_internal_imports()
    discovered = {_dependency_group(_module_name(path)) for path in _PACKAGE_ROOT.rglob("*.py")}
    missing = sorted(discovered - ALLOWED_DEPENDENCIES.keys())
    stale = sorted(ALLOWED_DEPENDENCIES.keys() - discovered)
    unknown_targets = sorted(
        {
            target
            for targets in ALLOWED_DEPENDENCIES.values()
            for target in targets
            if target not in discovered
        }
    )
    self_edges = sorted(
        source for source, targets in ALLOWED_DEPENDENCIES.items() if source in targets
    )
    observed_edges = {
        (_dependency_group(item.source_module), _dependency_group(item.target_module))
        for item in imports
        if item.target_module != _UNRESOLVED_DYNAMIC_MODULE
        and _dependency_group(item.source_module) != _dependency_group(item.target_module)
    }
    declared_edges = {
        (source, target) for source, targets in ALLOWED_DEPENDENCIES.items() for target in targets
    }
    unused_edges = sorted(declared_edges - observed_edges)

    assert (
        not missing and not stale and not unknown_targets and not self_edges and not unused_edges
    ), (
        "ALLOWED_DEPENDENCIES must describe only current, reviewed direct edges. "
        f"Missing policy: {missing or 'none'}; stale policy: {stale or 'none'}; "
        f"unknown targets: {unknown_targets or 'none'}; self edges: {self_edges or 'none'}; "
        f"pre-approved but unused edges: {unused_edges or 'none'}."
    )


def test_dependency_policy_is_acyclic_and_keeps_hosts_at_the_edge() -> None:
    """Keep adapters out of reusable packages and the reviewed graph cycle-free."""

    host_groups = {"cli", "mcp", "viewer", "web"}
    imports_of_hosts = sorted(
        (source, target)
        for source, targets in ALLOWED_DEPENDENCIES.items()
        for target in targets
        if target in host_groups
    )

    # Repeatedly remove packages whose dependencies have already bottomed out.
    # Any nodes left at the end participate in, or depend on, a cycle.
    remaining = {source: set(targets) for source, targets in ALLOWED_DEPENDENCIES.items()}
    while leaves := {source for source, targets in remaining.items() if not targets}:
        remaining = {
            source: targets - leaves
            for source, targets in remaining.items()
            if source not in leaves
        }

    assert not imports_of_hosts and not remaining, (
        "Reusable packages and sibling hosts must not import a delivery host; use the exact "
        "CLI launcher exceptions only at composition roots. The declared dependency graph must "
        f"also remain acyclic. Host edges: {imports_of_hosts or 'none'}; cycle remainder: "
        f"{remaining or 'none'}."
    )


def test_cli_host_launch_exceptions_are_exact_and_current() -> None:
    imports = _scan_internal_imports()
    source_paths = {
        path.relative_to(_REPOSITORY_ROOT).as_posix() for path in _PACKAGE_ROOT.rglob("*.py")
    }
    module_names = {_module_name(path) for path in _PACKAGE_ROOT.rglob("*.py")}
    configured = {
        (source_path, target_module)
        for source_path, targets in CLI_HOST_LAUNCH_EXCEPTIONS.items()
        for target_module in targets
    }
    observed = {(item.source_path, item.target_module) for item in imports}
    stale_sources = sorted(set(CLI_HOST_LAUNCH_EXCEPTIONS) - source_paths)
    unknown_targets = sorted(target for _, target in configured if target not in module_names)
    invalid_directions = sorted(
        (source, target)
        for source, target in configured
        if not source.startswith("hhtools/cli/")
        or _dependency_group(target) not in {"viewer", "web"}
    )
    unused = sorted(configured - observed)

    assert not stale_sources and not unknown_targets and not invalid_directions and not unused, (
        "CLI launch exceptions must identify existing source files and exact imported "
        "host modules. "
        f"Stale sources: {stale_sources or 'none'}; unknown targets: "
        f"{unknown_targets or 'none'}; invalid directions: {invalid_directions or 'none'}; "
        f"unused exceptions: {unused or 'none'}."
    )


def test_legacy_baseline_configuration_is_exact() -> None:
    location_keys = {
        (item.source_path, item.target_module, item.scope) for item in _LEGACY_LOCATION_BASELINE
    }
    fingerprint_keys = set(_LEGACY_IMPORT_FINGERPRINTS)
    missing_fingerprints = sorted(location_keys - fingerprint_keys)
    stale_fingerprints = sorted(fingerprint_keys - location_keys)
    invalid_counts = sorted(
        item
        for item, count in LEGACY_VIOLATION_BASELINE.items()
        if type(count) is not int or count <= 0
    )
    blank_fingerprints = sorted(item for item in LEGACY_VIOLATION_BASELINE if not item.fingerprint)

    assert (
        not missing_fingerprints
        and not stale_fingerprints
        and not invalid_counts
        and not blank_fingerprints
    ), (
        "The legacy snapshot must have one non-empty fingerprint and a positive count per debt "
        f"identity. Missing fingerprints: {missing_fingerprints or 'none'}; stale fingerprints: "
        f"{stale_fingerprints or 'none'}; invalid counts: {invalid_counts or 'none'}; blank "
        f"fingerprints: {blank_fingerprints or 'none'}."
    )


def test_import_scanner_sees_relative_lazy_and_dynamic_imports() -> None:
    imports = _scan_source(
        """
from typing import TYPE_CHECKING
from .. import viewer
import importlib as module_loader
from importlib import import_module as load_module

if TYPE_CHECKING:
    from hhtools.web import server

def lazy_load():
    from hhtools.mcp import runtime
    module_loader.import_module("hhtools.cli.main")
    load_module("hhtools.analysis.clip")
    __import__("hhtools.services.jobs")
"""
    )

    assert {(item.target_module, item.scope, item.fingerprint) for item in imports} == {
        ("hhtools.viewer", _MODULE_SCOPE, "from:viewer"),
        ("hhtools.web", _MODULE_SCOPE, "from:server"),
        ("hhtools.mcp", "lazy_load", "from:runtime"),
        ("hhtools.cli.main", "lazy_load", "dynamic:importlib.import_module"),
        ("hhtools.analysis.clip", "lazy_load", "dynamic:importlib.import_module"),
        ("hhtools.services.jobs", "lazy_load", "dynamic:__import__"),
    }


def test_import_scanner_fingerprints_symbols_without_order_noise() -> None:
    one_symbol = _scan_source("from hhtools.web.server import create_app")
    two_symbols = _scan_source("from hhtools.web.server import run_web, create_app")

    assert one_symbol[0].fingerprint == "from:create_app"
    assert two_symbols[0].fingerprint == "from:create_app,run_web"
    assert one_symbol[0].identity != two_symbols[0].identity


def test_import_scanner_handles_keyword_relative_and_constant_dynamic_names() -> None:
    imports = _scan_source(
        """
import importlib.util

importlib.import_module(name="hhtools." + "web")
importlib.import_module(".viewer", package="hhtools")
__import__(name="hhtools.services.jobs")

def unresolved(module_name):
    importlib.import_module(module_name)
"""
    )

    assert {(item.target_module, item.scope, item.fingerprint) for item in imports} == {
        ("hhtools.web", _MODULE_SCOPE, "dynamic:importlib.import_module"),
        ("hhtools.viewer", _MODULE_SCOPE, "dynamic:importlib.import_module"),
        ("hhtools.services.jobs", _MODULE_SCOPE, "dynamic:__import__"),
        (
            _UNRESOLVED_DYNAMIC_MODULE,
            "unresolved",
            "dynamic:importlib.import_module:module_name",
        ),
    }


def test_import_scanner_rejects_unresolvable_static_relative_imports() -> None:
    with pytest.raises(AssertionError, match="Cannot resolve relative import"):
        _scan_source("from ....web import server")


def test_python_package_dependencies_do_not_regress() -> None:
    forbidden = [item for item in _scan_internal_imports() if not _is_allowed(item)]
    occurrences: dict[ImportIdentity, list[InternalImport]] = defaultdict(list)
    for item in forbidden:
        occurrences[item.identity].append(item)

    actual = Counter({identity: len(items) for identity, items in occurrences.items()})
    baseline = Counter(LEGACY_VIOLATION_BASELINE)
    regressions = actual - baseline
    removed = baseline - actual

    assert not regressions, _format_regressions(regressions, occurrences)
    assert not removed, _format_removed_debt(removed)
