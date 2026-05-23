from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_GOLDEN_CAPTURE_PATTERNS = (
    "workflow_manifest.json",
    "iter_*/iteration_state.json",
    "iter_*/design_cage.npz",
    "iter_*/cage_3d.npz",
    "iter_*/mma_state.npz",
    "iter_*/aggregate_terms.npz",
    "iter_*/*/case_summary.json",
    "iter_*/*/forward_t*/*.npz",
    "iter_*/*/adjoint_t*/*.npz",
    "iter_*/*/adjoint_t*/fv_cload.inp",
    "iter_*/*/adjoint_t*/fv_manifest.json",
    "**/*.inp",
    "**/U1.txt",
    "**/*_U1_vectors.npz",
    "**/job_metadata.json",
)


MMA_STATE_KEYS = (
    "iteration",
    "xold1",
    "xold2",
    "xmin",
    "xmax",
    "low",
    "up",
    "a0",
    "a",
    "c",
    "d",
)
AGGREGATE_KEYS = ("objective", "d_ob", "g2", "d_g2")
CASE_HISTORY_KEYS = ("obj_bo_history", "bo_sum_history", "fai_history", "initial_design_sensitivity")
FORWARD_STEP_KEYS = (
    "full_element_displacements",
    "obj_bo_previous",
    "obj_bo_next",
    "bone_s",
    "bone_density_delta",
    "design_sensitivity",
)
FV_KEYS = ("nodal_forces_flat", "active_node_ids", "active_forces_xyz")
FAI_KEYS = ("fai_next", "fai_current", "adjoint_element_displacements")


@dataclass(frozen=True, slots=True)
class FJWValidationCheck:
    name: str
    status: str
    message: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FJWGoldenCaptureRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    copied: bool
    copied_path: str | None = None

    def as_jsonable(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "copied": self.copied,
            "copied_path": self.copied_path,
        }


@dataclass(frozen=True, slots=True)
class FJWGoldenCaptureReport:
    source_run_directory: Path
    golden_directory: Path
    manifest_path: Path
    records: tuple[FJWGoldenCaptureRecord, ...]

    def as_jsonable(self) -> dict[str, object]:
        return {
            "source_run_directory": str(self.source_run_directory),
            "golden_directory": str(self.golden_directory),
            "manifest_path": str(self.manifest_path),
            "file_count": len(self.records),
            "copied_file_count": sum(1 for record in self.records if record.copied),
            "records": [record.as_jsonable() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class FJWValidationReport:
    run_directory: Path
    status: str
    checks: tuple[FJWValidationCheck, ...]
    historical_equivalence_claim: str

    def as_jsonable(self) -> dict[str, object]:
        return {
            "run_directory": str(self.run_directory),
            "status": self.status,
            "historical_equivalence_claim": self.historical_equivalence_claim,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "metadata": check.metadata,
                }
                for check in self.checks
            ],
        }


def normalize_inp_text(text: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"\s*,\s*", ",", stripped)
        stripped = re.sub(r"\s+", " ", stripped)
        normalized_lines.append(stripped.lower())
    return "\n".join(normalized_lines)


def compare_text_files(actual_path: Path, expected_path: Path) -> FJWValidationCheck:
    actual = normalize_inp_text(Path(actual_path).read_text(encoding="utf-8"))
    expected = normalize_inp_text(Path(expected_path).read_text(encoding="utf-8"))
    if actual == expected:
        return FJWValidationCheck(
            name=f"text:{Path(actual_path).name}",
            status="pass",
            message="normalized text matches",
            metadata={"actual": str(actual_path), "expected": str(expected_path)},
        )
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    mismatch_index = next(
        (
            index
            for index, (left, right) in enumerate(zip(actual_lines, expected_lines, strict=False))
            if left != right
        ),
        min(len(actual_lines), len(expected_lines)),
    )
    return FJWValidationCheck(
        name=f"text:{Path(actual_path).name}",
        status="fail",
        message="normalized text differs",
        metadata={
            "actual": str(actual_path),
            "expected": str(expected_path),
            "first_mismatch_line": int(mismatch_index + 1),
        },
    )


def compare_npz_arrays(
    actual_path: Path,
    expected_path: Path,
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> tuple[FJWValidationCheck, ...]:
    actual = np.load(actual_path)
    expected = np.load(expected_path)
    checks: list[FJWValidationCheck] = []
    actual_keys = set(actual.files)
    expected_keys = set(expected.files)
    if actual_keys != expected_keys:
        checks.append(
            FJWValidationCheck(
                name=f"npz:{Path(actual_path).name}:keys",
                status="fail",
                message="array keys differ",
                metadata={
                    "actual_only": sorted(actual_keys - expected_keys),
                    "expected_only": sorted(expected_keys - actual_keys),
                },
            )
        )
        return tuple(checks)

    for key in sorted(actual_keys):
        actual_array = np.asarray(actual[key])
        expected_array = np.asarray(expected[key])
        if actual_array.shape != expected_array.shape:
            checks.append(
                FJWValidationCheck(
                    name=f"npz:{Path(actual_path).name}:{key}",
                    status="fail",
                    message="array shape differs",
                    metadata={"actual_shape": actual_array.shape, "expected_shape": expected_array.shape},
                )
            )
            continue
        diff = np.abs(actual_array - expected_array)
        max_error = float(np.max(diff)) if diff.size else 0.0
        status = "pass" if np.allclose(actual_array, expected_array, rtol=rtol, atol=atol) else "fail"
        checks.append(
            FJWValidationCheck(
                name=f"npz:{Path(actual_path).name}:{key}",
                status=status,
                message="array comparison complete",
                metadata={"max_error": max_error, "rtol": rtol, "atol": atol},
            )
        )
    return tuple(checks)


def capture_fjw_golden_run(
    run_directory: Path,
    golden_directory: Path,
    *,
    copy_max_bytes: int = 5_000_000,
    copy_files: bool = True,
    require_valid_run: bool = True,
    include_patterns: tuple[str, ...] = DEFAULT_GOLDEN_CAPTURE_PATTERNS,
) -> FJWGoldenCaptureReport:
    root = Path(run_directory)
    if not root.exists():
        raise FileNotFoundError(f"FJW run directory does not exist: {root}")
    if copy_max_bytes < 0:
        raise ValueError("copy_max_bytes must be non-negative.")

    source_validation = validate_run_directory(root)
    failed_source_checks = tuple(check for check in source_validation.checks if check.status == "fail")
    if require_valid_run and failed_source_checks:
        failed_names = ", ".join(check.name for check in failed_source_checks)
        raise ValueError(f"Cannot capture FJW golden from invalid run directory; failed checks: {failed_names}")

    target_root = Path(golden_directory)
    target_root.mkdir(parents=True, exist_ok=True)
    records: list[FJWGoldenCaptureRecord] = []
    for path in _iter_capture_files(root, include_patterns):
        relative = path.relative_to(root).as_posix()
        size_bytes = path.stat().st_size
        copied = bool(copy_files and size_bytes <= copy_max_bytes)
        copied_path = None
        if copied:
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied_path = relative
        records.append(
            FJWGoldenCaptureRecord(
                relative_path=relative,
                size_bytes=int(size_bytes),
                sha256=_sha256_file(path),
                copied=copied,
                copied_path=copied_path,
            )
        )

    manifest_path = target_root / "golden_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_directory": str(root.resolve()),
        "source_validation_status": source_validation.status,
        "copy_max_bytes": int(copy_max_bytes),
        "include_patterns": list(include_patterns),
        "files": [record.as_jsonable() for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return FJWGoldenCaptureReport(
        source_run_directory=root,
        golden_directory=target_root,
        manifest_path=manifest_path,
        records=tuple(records),
    )


def compare_golden_manifest(run_directory: Path, golden_manifest_path: Path) -> tuple[FJWValidationCheck, ...]:
    root = Path(run_directory)
    manifest_path = Path(golden_manifest_path)
    if not manifest_path.exists():
        return (
            FJWValidationCheck(
                name="golden_manifest",
                status="fail",
                message="golden manifest does not exist",
                metadata={"path": str(manifest_path)},
            ),
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[FJWValidationCheck] = []
    for record in manifest.get("files", []):
        relative = Path(str(record["relative_path"]))
        actual_path = root / relative
        check_name = f"golden_manifest:{relative.as_posix()}"
        if not actual_path.exists():
            checks.append(
                FJWValidationCheck(
                    name=check_name,
                    status="fail",
                    message="actual file is missing",
                    metadata={"actual": str(actual_path), "manifest": str(manifest_path)},
                )
            )
            continue

        actual_size = actual_path.stat().st_size
        expected_size = int(record["size_bytes"])
        if actual_size != expected_size:
            checks.append(
                FJWValidationCheck(
                    name=check_name,
                    status="fail",
                    message="file size differs from golden manifest",
                    metadata={"actual_size": int(actual_size), "expected_size": expected_size},
                )
            )
            continue

        actual_sha = _sha256_file(actual_path)
        expected_sha = str(record["sha256"])
        checks.append(
            FJWValidationCheck(
                name=check_name,
                status="pass" if actual_sha == expected_sha else "fail",
                message="checksum comparison complete",
                metadata={
                    "actual_sha256": actual_sha,
                    "expected_sha256": expected_sha,
                    "size_bytes": expected_size,
                },
            )
        )
    if not checks:
        checks.append(
            FJWValidationCheck(
                name="golden_manifest",
                status="fail",
                message="golden manifest contains no files",
                metadata={"path": str(manifest_path)},
            )
        )
    return tuple(checks)


def validate_run_directory(
    run_directory: Path,
    *,
    golden_directory: Path | None = None,
) -> FJWValidationReport:
    root = Path(run_directory)
    checks: list[FJWValidationCheck] = []
    checks.append(_check_exists(root / "workflow_manifest.json", "workflow_manifest"))
    iteration_dirs = tuple(sorted(path for path in root.glob("iter_*") if path.is_dir()))
    if iteration_dirs:
        checks.append(
            FJWValidationCheck(
                name="iteration_checkpoints",
                status="pass",
                message="iteration checkpoint directories found",
                metadata={"count": len(iteration_dirs), "directories": [str(path) for path in iteration_dirs]},
            )
        )
    else:
        checks.append(
            FJWValidationCheck(
                name="iteration_checkpoints",
                status="fail",
                message="no iter_* checkpoint directories found",
            )
        )

    for checkpoint in iteration_dirs:
        checks.extend(_validate_checkpoint_files(checkpoint))

    if golden_directory is None:
        checks.append(
            FJWValidationCheck(
                name="historical_golden_data",
                status="warn",
                message="no golden directory was provided; historical MATLAB+Abaqus equivalence cannot be proven",
            )
        )
    else:
        checks.extend(_compare_against_golden(root, Path(golden_directory)))

    status = "pass"
    if any(check.status == "fail" for check in checks):
        status = "fail"
    elif any(check.status == "warn" for check in checks):
        status = "warn"
    claim = (
        "validated_against_golden"
        if golden_directory is not None and status == "pass"
        else "cannot_prove_historical_equivalence_without_matching_golden_outputs"
    )
    return FJWValidationReport(
        run_directory=root,
        status=status,
        checks=tuple(checks),
        historical_equivalence_claim=claim,
    )


def write_validation_report(
    report: FJWValidationReport,
    output_path: Path | None = None,
) -> Path:
    path = Path(output_path) if output_path is not None else report.run_directory / "validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_jsonable(), indent=2), encoding="utf-8")
    return path


def _check_exists(path: Path, name: str) -> FJWValidationCheck:
    if path.exists():
        return FJWValidationCheck(name=name, status="pass", message="file exists", metadata={"path": str(path)})
    return FJWValidationCheck(name=name, status="fail", message="missing file", metadata={"path": str(path)})


def _validate_checkpoint_files(checkpoint: Path) -> tuple[FJWValidationCheck, ...]:
    required = (
        "iteration_state.json",
        "design_cage.npz",
        "mma_state.npz",
    )
    checks: list[FJWValidationCheck] = [
        _check_exists(checkpoint / filename, f"{checkpoint.name}:{filename}") for filename in required
    ]
    iteration_payload = _read_json_payload(checkpoint / "iteration_state.json", f"{checkpoint.name}:iteration_state.json")
    checks.extend(iteration_payload[0])
    iteration_state = iteration_payload[1]
    checks.extend(
        _validate_npz_payload(
            checkpoint / "design_cage.npz",
            f"{checkpoint.name}:design_cage.npz",
            required_keys=("design_cage",),
            finite_keys=("design_cage", "next_design"),
        )
    )
    checks.extend(
        _validate_npz_payload(
            checkpoint / "mma_state.npz",
            f"{checkpoint.name}:mma_state.npz",
            required_keys=MMA_STATE_KEYS,
            finite_keys=MMA_STATE_KEYS,
        )
    )
    if iteration_state.get("checkpoint_kind") == "initial":
        return tuple(checks)

    checks.extend(_validate_completed_iteration_state(checkpoint, iteration_state))
    checks.extend(
        _validate_npz_payload(
            checkpoint / "aggregate_terms.npz",
            f"{checkpoint.name}:aggregate_terms.npz",
            required_keys=AGGREGATE_KEYS,
            finite_keys=AGGREGATE_KEYS,
        )
    )
    checks.extend(_validate_case_artifacts(checkpoint, iteration_state))
    return tuple(checks)


def _read_json_payload(path: Path, name: str) -> tuple[tuple[FJWValidationCheck, ...], dict[str, object]]:
    if not path.exists():
        return (
            (
                FJWValidationCheck(
                    name=name,
                    status="fail",
                    message="missing JSON file",
                    metadata={"path": str(path)},
                ),
            ),
            {},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            (
                FJWValidationCheck(
                    name=name,
                    status="fail",
                    message="invalid JSON file",
                    metadata={"path": str(path), "error": str(exc)},
                ),
            ),
            {},
        )
    if not isinstance(payload, dict):
        return (
            (
                FJWValidationCheck(
                    name=name,
                    status="fail",
                    message="JSON payload must be an object",
                    metadata={"path": str(path)},
                ),
            ),
            {},
        )
    return (
        (
            FJWValidationCheck(
                name=name,
                status="pass",
                message="JSON file is readable",
                metadata={"path": str(path)},
            ),
        ),
        payload,
    )


def _validate_completed_iteration_state(
    checkpoint: Path,
    iteration_state: dict[str, object],
) -> tuple[FJWValidationCheck, ...]:
    checks: list[FJWValidationCheck] = []
    has_placeholder = bool(iteration_state.get("has_placeholder_adjoint", False))
    checks.append(
        FJWValidationCheck(
            name=f"{checkpoint.name}:has_placeholder_adjoint",
            status="fail" if has_placeholder else "pass",
            message="placeholder adjoint flag checked",
            metadata={"value": has_placeholder},
        )
    )
    for key in ("delta", "objective", "g2"):
        if key not in iteration_state or iteration_state[key] is None:
            checks.append(
                FJWValidationCheck(
                    name=f"{checkpoint.name}:iteration_state:{key}",
                    status="fail",
                    message="required scalar is missing",
                )
            )
            continue
        value = float(iteration_state[key])
        checks.append(
            FJWValidationCheck(
                name=f"{checkpoint.name}:iteration_state:{key}",
                status="pass" if np.isfinite(value) else "fail",
                message="required scalar is finite",
                metadata={"value": value},
            )
        )
    cases = iteration_state.get("cases")
    checks.append(
        FJWValidationCheck(
            name=f"{checkpoint.name}:cases",
            status="pass" if isinstance(cases, list) and len(cases) > 0 else "fail",
            message="case summaries are present",
            metadata={"count": 0 if not isinstance(cases, list) else len(cases)},
        )
    )
    return tuple(checks)


def _validate_case_artifacts(
    checkpoint: Path,
    iteration_state: dict[str, object],
) -> tuple[FJWValidationCheck, ...]:
    checks: list[FJWValidationCheck] = []
    cases = iteration_state.get("cases")
    if not isinstance(cases, list):
        return tuple(checks)
    for case in cases:
        if not isinstance(case, dict):
            checks.append(
                FJWValidationCheck(
                    name=f"{checkpoint.name}:case",
                    status="fail",
                    message="case summary must be an object",
                )
            )
            continue
        case_name = str(case.get("load_case_name", ""))
        case_dir = checkpoint / case_name
        checks.append(_check_exists(case_dir, f"{checkpoint.name}:{case_name}"))
        checks.extend(
            _validate_npz_payload(
                case_dir / "case_history.npz",
                f"{checkpoint.name}:{case_name}:case_history.npz",
                required_keys=CASE_HISTORY_KEYS,
                finite_keys=CASE_HISTORY_KEYS,
            )
        )
        forward_count = int(case.get("forward_step_count", 0))
        adjoint_count = int(case.get("adjoint_step_count", 0))
        checks.extend(_validate_forward_steps(case_dir, forward_count))
        checks.extend(_validate_adjoint_steps(case_dir, adjoint_count))
    return tuple(checks)


def _validate_forward_steps(case_dir: Path, expected_count: int) -> tuple[FJWValidationCheck, ...]:
    checks: list[FJWValidationCheck] = []
    step_dirs = tuple(sorted(path for path in case_dir.glob("forward_t*") if path.is_dir()))
    checks.append(
        FJWValidationCheck(
            name=f"{case_dir.parent.name}:{case_dir.name}:forward_steps",
            status="pass" if len(step_dirs) == expected_count else "fail",
            message="forward step count checked",
            metadata={"actual": len(step_dirs), "expected": expected_count},
        )
    )
    for step_dir in step_dirs:
        checks.extend(
            _validate_npz_payload(
                step_dir / "forward_step.npz",
                f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:forward_step.npz",
                required_keys=FORWARD_STEP_KEYS,
                finite_keys=FORWARD_STEP_KEYS,
                nonzero_keys=("full_element_displacements",),
            )
        )
    return tuple(checks)


def _validate_adjoint_steps(case_dir: Path, expected_count: int) -> tuple[FJWValidationCheck, ...]:
    checks: list[FJWValidationCheck] = []
    step_dirs = tuple(sorted(path for path in case_dir.glob("adjoint_t*") if path.is_dir()))
    checks.append(
        FJWValidationCheck(
            name=f"{case_dir.parent.name}:{case_dir.name}:adjoint_steps",
            status="pass" if len(step_dirs) == expected_count else "fail",
            message="adjoint step count checked",
            metadata={"actual": len(step_dirs), "expected": expected_count},
        )
    )
    for step_dir in step_dirs:
        checks.extend(
            _validate_npz_payload(
                step_dir / "fv.npz",
                f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fv.npz",
                required_keys=FV_KEYS,
                finite_keys=("nodal_forces_flat", "active_forces_xyz"),
            )
        )
        checks.extend(
            _validate_npz_payload(
                step_dir / "fai.npz",
                f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fai.npz",
                required_keys=FAI_KEYS,
                finite_keys=FAI_KEYS,
                nonzero_keys=("adjoint_element_displacements",),
            )
        )
        checks.append(_check_exists(step_dir / "fv_cload.inp", f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fv_cload.inp"))
        manifest_checks, manifest = _read_json_payload(
            step_dir / "fv_manifest.json",
            f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fv_manifest.json",
        )
        checks.extend(manifest_checks)
        if manifest:
            active_count = int(manifest.get("active_node_count", 0))
            dense_l2_norm = float(manifest.get("dense_l2_norm", 0.0))
            checks.append(
                FJWValidationCheck(
                    name=f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fv_manifest:active_node_count",
                    status="pass" if active_count > 0 else "warn",
                    message="dynamic Fv active node count checked",
                    metadata={"active_node_count": active_count},
                )
            )
            checks.append(
                FJWValidationCheck(
                    name=f"{case_dir.parent.name}:{case_dir.name}:{step_dir.name}:fv_manifest:dense_l2_norm",
                    status="pass" if dense_l2_norm > 0.0 else "warn",
                    message="dynamic Fv vector norm checked",
                    metadata={"dense_l2_norm": dense_l2_norm},
                )
            )
    return tuple(checks)


def _validate_npz_payload(
    path: Path,
    name: str,
    *,
    required_keys: tuple[str, ...],
    finite_keys: tuple[str, ...] = (),
    nonzero_keys: tuple[str, ...] = (),
) -> tuple[FJWValidationCheck, ...]:
    if not path.exists():
        return (
            FJWValidationCheck(
                name=name,
                status="fail",
                message="missing NPZ file",
                metadata={"path": str(path)},
            ),
        )
    try:
        with np.load(path) as payload:
            keys = set(payload.files)
            checks: list[FJWValidationCheck] = []
            missing = tuple(key for key in required_keys if key not in keys)
            checks.append(
                FJWValidationCheck(
                    name=f"{name}:keys",
                    status="pass" if not missing else "fail",
                    message="required NPZ keys checked",
                    metadata={"missing": list(missing), "keys": sorted(keys)},
                )
            )
            for key in finite_keys:
                if key not in keys:
                    continue
                array = np.asarray(payload[key])
                is_finite = bool(np.all(np.isfinite(array)))
                checks.append(
                    FJWValidationCheck(
                        name=f"{name}:{key}:finite",
                        status="pass" if is_finite else "fail",
                        message="array finiteness checked",
                        metadata={"shape": list(array.shape)},
                    )
                )
            for key in nonzero_keys:
                if key not in keys:
                    continue
                array = np.asarray(payload[key])
                max_abs = float(np.max(np.abs(array))) if array.size else 0.0
                checks.append(
                    FJWValidationCheck(
                        name=f"{name}:{key}:nonzero",
                        status="pass" if max_abs > 0.0 else "warn",
                        message="array nonzero check helps detect placeholder solve outputs",
                        metadata={"max_abs": max_abs},
                    )
                )
            return tuple(checks)
    except Exception as exc:
        return (
            FJWValidationCheck(
                name=name,
                status="fail",
                message="NPZ file could not be read",
                metadata={"path": str(path), "error": str(exc)},
            ),
        )


def _compare_against_golden(run_directory: Path, golden_directory: Path) -> Iterable[FJWValidationCheck]:
    if not golden_directory.exists():
        return (
            FJWValidationCheck(
                name="golden_directory",
                status="fail",
                message="golden directory does not exist",
                metadata={"path": str(golden_directory)},
            ),
        )
    checks: list[FJWValidationCheck] = []
    manifest_path = golden_directory / "golden_manifest.json"
    if manifest_path.exists():
        checks.extend(compare_golden_manifest(run_directory, manifest_path))
    for expected_path in golden_directory.rglob("*"):
        if not expected_path.is_file():
            continue
        if expected_path.name == "golden_manifest.json":
            continue
        relative = expected_path.relative_to(golden_directory)
        actual_path = run_directory / relative
        if not actual_path.exists():
            checks.append(
                FJWValidationCheck(
                    name=f"golden:{relative}",
                    status="fail",
                    message="actual file is missing",
                    metadata={"actual": str(actual_path), "expected": str(expected_path)},
                )
            )
            continue
        if expected_path.suffix == ".npz":
            checks.extend(compare_npz_arrays(actual_path, expected_path))
        elif expected_path.suffix in {".inp", ".txt"}:
            checks.append(compare_text_files(actual_path, expected_path))
    return tuple(checks)


def _iter_capture_files(root: Path, patterns: tuple[str, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_GOLDEN_CAPTURE_PATTERNS",
    "FJWGoldenCaptureRecord",
    "FJWGoldenCaptureReport",
    "FJWValidationCheck",
    "FJWValidationReport",
    "capture_fjw_golden_run",
    "compare_golden_manifest",
    "compare_npz_arrays",
    "compare_text_files",
    "normalize_inp_text",
    "validate_run_directory",
    "write_validation_report",
]
