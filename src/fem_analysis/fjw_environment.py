from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FJWPreflightStatus = Literal["pass", "warn", "fail"]


_RUNTIME_GOLDEN_PATTERNS = (
    "*.odb",
    "Force_*.mat",
    "U1_ele_nod_dir*.mat",
    "obj_bo*.mat",
    "ob*.mat",
)
_STATIC_REFERENCE_NAMES = {"obj_ele.mat"}


@dataclass(frozen=True, slots=True)
class FJWPreflightCheck:
    name: str
    status: FJWPreflightStatus
    message: str
    metadata: dict[str, object] = field(default_factory=dict)

    def as_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class FJWPreflightReport:
    checks: tuple[FJWPreflightCheck, ...]

    @property
    def status(self) -> FJWPreflightStatus:
        if any(check.status == "fail" for check in self.checks):
            return "fail"
        if any(check.status == "warn" for check in self.checks):
            return "warn"
        return "pass"

    @property
    def is_success(self) -> bool:
        return self.status != "fail"

    def as_jsonable(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checks": [check.as_jsonable() for check in self.checks],
        }


def find_fjw_runtime_golden_outputs(reference_dir: Path) -> tuple[Path, ...]:
    root = Path(reference_dir)
    if not root.exists():
        return ()

    paths: dict[Path, Path] = {}
    for pattern in _RUNTIME_GOLDEN_PATTERNS:
        for path in root.rglob(pattern):
            if path.name in _STATIC_REFERENCE_NAMES:
                continue
            if path.is_file():
                paths[path.resolve()] = path
    return tuple(sorted(paths.values(), key=lambda item: str(item)))


def check_fjw_runtime_environment(
    *,
    reference_dir: Path,
    golden_directory: Path | None = None,
    abaqus_executable: str = "abaqus",
    require_abaqus: bool = False,
    require_petsc_mumps: bool = False,
    require_golden: bool = False,
) -> FJWPreflightReport:
    reference_root = Path(reference_dir)
    checks: list[FJWPreflightCheck] = []

    if reference_root.exists():
        checks.append(
            FJWPreflightCheck(
                name="reference_dir",
                status="pass",
                message="reference directory exists",
                metadata={"path": str(reference_root.resolve())},
            )
        )
    else:
        checks.append(
            FJWPreflightCheck(
                name="reference_dir",
                status="fail",
                message="reference directory is missing",
                metadata={"path": str(reference_root)},
            )
        )

    checks.extend(_check_python_solver_stack())

    if require_abaqus:
        checks.append(_check_abaqus_executable(abaqus_executable))

    if require_petsc_mumps:
        checks.append(_check_petsc_mumps_runtime(require_petsc_mumps=True))

    golden_sources = _find_golden_sources(reference_root, golden_directory)
    if golden_sources:
        checks.append(
            FJWPreflightCheck(
                name="runtime_golden_outputs",
                status="pass",
                message="historical or captured runtime golden source was found",
                metadata={
                    "count": len(golden_sources),
                    "paths": [str(path) for path in golden_sources[:20]],
                    "truncated": len(golden_sources) > 20,
                },
            )
        )
    elif require_golden:
        checks.append(
            FJWPreflightCheck(
                name="runtime_golden_outputs",
                status="fail",
                message="no historical runtime golden outputs were found",
                metadata={
                    "required": require_golden,
                    "golden_directory": None if golden_directory is None else str(golden_directory),
                    "patterns": list(_RUNTIME_GOLDEN_PATTERNS),
                    "captured_manifest": "golden_manifest.json",
                    "excluded_static_files": sorted(_STATIC_REFERENCE_NAMES),
                },
            )
        )

    return FJWPreflightReport(checks=tuple(checks))


def write_fjw_preflight_report(
    report: FJWPreflightReport,
    *,
    output_path: Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_jsonable(), indent=2), encoding="utf-8")
    return path


def _resolve_executable(executable: str) -> Path | None:
    candidate = Path(executable)
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(executable)
    return None if resolved is None else Path(resolved).resolve()


def _check_python_solver_stack() -> tuple[FJWPreflightCheck, ...]:
    checks: list[FJWPreflightCheck] = []
    for package_name in ("numpy", "scipy", "sfepy"):
        spec = importlib.util.find_spec(package_name)
        checks.append(
            FJWPreflightCheck(
                name=f"python_dependency:{package_name}",
                status="pass" if spec is not None else "fail",
                message=f"{package_name} is importable" if spec is not None else f"{package_name} is not importable",
                metadata={"origin": None if spec is None else spec.origin},
            )
        )
    return tuple(checks)


def _check_abaqus_executable(abaqus_executable: str) -> FJWPreflightCheck:
    abaqus_path = _resolve_executable(abaqus_executable)
    if abaqus_path is None:
        return FJWPreflightCheck(
            name="abaqus_executable",
            status="fail",
            message="Abaqus executable was not found",
            metadata={"requested": abaqus_executable, "required": True},
        )
    return FJWPreflightCheck(
        name="abaqus_executable",
        status="pass",
        message="Abaqus executable is available",
        metadata={"requested": abaqus_executable, "resolved": str(abaqus_path)},
    )


def _find_golden_sources(reference_dir: Path, golden_directory: Path | None) -> tuple[Path, ...]:
    paths = list(find_fjw_runtime_golden_outputs(reference_dir))
    if golden_directory is not None:
        captured_root = Path(golden_directory)
        manifest = captured_root / "golden_manifest.json"
        if manifest.exists():
            paths.append(manifest)
        paths.extend(find_fjw_runtime_golden_outputs(captured_root))
    unique: dict[Path, Path] = {}
    for path in paths:
        unique[path.resolve()] = path
    return tuple(sorted(unique.values(), key=lambda item: str(item)))


def _check_petsc_mumps_runtime(*, require_petsc_mumps: bool) -> FJWPreflightCheck:
    petsc4py_spec = importlib.util.find_spec("petsc4py")
    if petsc4py_spec is None:
        return FJWPreflightCheck(
            name="petsc_mumps_runtime",
            status="fail" if require_petsc_mumps else "warn",
            message="petsc4py is not importable; PETSc/MUMPS full-model validation cannot run",
            metadata={"required": require_petsc_mumps},
        )

    metadata: dict[str, object] = {"required": require_petsc_mumps, "origin": petsc4py_spec.origin}
    try:
        petsc = importlib.import_module("petsc4py.PETSc")
    except Exception as exc:  # pragma: no cover - depends on external PETSc builds.
        return FJWPreflightCheck(
            name="petsc_mumps_runtime",
            status="fail" if require_petsc_mumps else "warn",
            message="petsc4py is present but PETSc could not be imported",
            metadata={**metadata, "error": str(exc)},
        )

    sys_api = getattr(petsc, "Sys", None)
    has_external = getattr(sys_api, "hasExternalPackage", None)
    if not callable(has_external):
        return FJWPreflightCheck(
            name="petsc_mumps_runtime",
            status="fail" if require_petsc_mumps else "warn",
            message="petsc4py is present but MUMPS support could not be inspected",
            metadata=metadata,
        )

    has_mumps = bool(has_external("mumps"))
    metadata["has_mumps"] = has_mumps
    if not has_mumps:
        return FJWPreflightCheck(
            name="petsc_mumps_runtime",
            status="fail" if require_petsc_mumps else "warn",
            message="petsc4py is importable but PETSc does not report MUMPS support",
            metadata=metadata,
        )

    return FJWPreflightCheck(
        name="petsc_mumps_runtime",
        status="pass",
        message="petsc4py is importable and PETSc reports MUMPS support",
        metadata=metadata,
    )


__all__ = [
    "FJWPreflightCheck",
    "FJWPreflightReport",
    "check_fjw_runtime_environment",
    "find_fjw_runtime_golden_outputs",
    "write_fjw_preflight_report",
]
