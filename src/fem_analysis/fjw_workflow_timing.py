from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _read_linux_process_memory() -> dict[str, int]:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return {}

    fields = {
        "VmRSS": "rss_bytes",
        "VmSize": "vms_bytes",
        "VmSwap": "swap_bytes",
        "Threads": "thread_count",
    }
    result: dict[str, int] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        key, _, raw_value = line.partition(":")
        output_key = fields.get(key)
        if output_key is None:
            continue
        parts = raw_value.strip().split()
        if not parts:
            continue
        value = int(parts[0])
        result[output_key] = value if key == "Threads" else value * 1024
    return result


def _process_status() -> dict[str, object]:
    user_time, system_time, children_user_time, children_system_time, elapsed_time = os.times()
    payload: dict[str, object] = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cpu_user_seconds": float(user_time),
        "cpu_system_seconds": float(system_time),
        "children_cpu_user_seconds": float(children_user_time),
        "children_cpu_system_seconds": float(children_system_time),
        "process_elapsed_seconds": float(elapsed_time),
    }
    try:
        payload["load_average"] = list(os.getloadavg())
    except OSError:
        pass
    payload.update(_read_linux_process_memory())
    return payload


@dataclass(slots=True)
class FJWTimingNode:
    name: str
    elapsed_seconds: float = 0.0
    children: list["FJWTimingNode"] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
            "children": [child.as_jsonable() for child in self.children],
        }


@dataclass(slots=True)
class _FJWActivePhase:
    token: int
    root_name: str
    name: str
    metadata: dict[str, object]
    started_at_unix: float
    started_perf_counter: float
    thread_name: str
    thread_id: int | None

    def as_jsonable(self, now_perf_counter: float) -> dict[str, object]:
        return {
            "token": self.token,
            "root_name": self.root_name,
            "name": self.name,
            "metadata": self.metadata,
            "started_at_unix": self.started_at_unix,
            "elapsed_seconds": max(0.0, now_perf_counter - self.started_perf_counter),
            "thread_name": self.thread_name,
            "thread_id": self.thread_id,
        }


@dataclass(slots=True)
class FJWHeartbeatWriter:
    status_path: Path
    events_path: Path | None = None
    interval_seconds: float = 30.0
    run_metadata: dict[str, object] = field(default_factory=dict)
    _lock: threading.Lock = field(init=False)
    _active_phases: dict[int, _FJWActivePhase] = field(init=False)
    _next_token: int = field(init=False, default=0)
    _run_state: str = field(init=False, default="initializing")
    _stop_event: threading.Event = field(init=False)
    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.status_path = Path(self.status_path)
        self.events_path = None if self.events_path is None else Path(self.events_path)
        if self.interval_seconds <= 0.0:
            raise ValueError("heartbeat interval_seconds must be positive.")
        self._lock = threading.Lock()
        self._active_phases = {}
        self._stop_event = threading.Event()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if self.events_path is not None:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="fjw-heartbeat",
            daemon=True,
        )
        self._thread.start()
        self.mark_run_state("initializing")

    def mark_run_state(self, state: str, **metadata: object) -> None:
        with self._lock:
            self._run_state = str(state)
            self._write_event_locked("run_state", state=str(state), metadata=_jsonable(metadata))
            self._write_status_locked("run_state")

    def start_phase(self, root_name: str, name: str, metadata: dict[str, object]) -> int:
        current_thread = threading.current_thread()
        with self._lock:
            self._next_token += 1
            token = self._next_token
            metadata_payload = _jsonable(metadata)
            if not isinstance(metadata_payload, dict):
                metadata_payload = {}
            phase = _FJWActivePhase(
                token=token,
                root_name=str(root_name),
                name=str(name),
                metadata=metadata_payload,
                started_at_unix=time.time(),
                started_perf_counter=time.perf_counter(),
                thread_name=current_thread.name,
                thread_id=current_thread.ident,
            )
            self._active_phases[token] = phase
            self._write_event_locked(
                "phase_start",
                token=token,
                root_name=phase.root_name,
                name=phase.name,
                metadata=phase.metadata,
                thread_name=phase.thread_name,
                thread_id=phase.thread_id,
            )
            self._write_status_locked("phase_start")
            return token

    def finish_phase(
        self,
        token: int,
        *,
        elapsed_seconds: float,
        error: str | None = None,
    ) -> None:
        with self._lock:
            phase = self._active_phases.pop(token, None)
            self._write_event_locked(
                "phase_end",
                token=token,
                root_name=None if phase is None else phase.root_name,
                name=None if phase is None else phase.name,
                elapsed_seconds=float(elapsed_seconds),
                status="error" if error else "ok",
                error=error,
            )
            self._write_status_locked("phase_end")

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=min(5.0, self.interval_seconds))
        with self._lock:
            self._write_status_locked("close")

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                if self._active_phases:
                    self._write_event_locked("heartbeat")
                self._write_status_locked("heartbeat")

    def _status_payload_locked(self, reason: str) -> dict[str, object]:
        now_perf_counter = time.perf_counter()
        return {
            "schema_version": 1,
            "timestamp_unix": time.time(),
            "reason": reason,
            "run_state": self._run_state,
            "run_metadata": _jsonable(self.run_metadata),
            "process": _process_status(),
            "active_phase_count": len(self._active_phases),
            "active_phases": [
                phase.as_jsonable(now_perf_counter)
                for phase in sorted(self._active_phases.values(), key=lambda item: item.token)
            ],
        }

    def _write_status_locked(self, reason: str) -> None:
        payload = self._status_payload_locked(reason)
        temp_path = self.status_path.with_suffix(f"{self.status_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.status_path)

    def _write_event_locked(self, event_type: str, **fields: object) -> None:
        if self.events_path is None:
            return
        payload = {
            "schema_version": 1,
            "timestamp_unix": time.time(),
            "event_type": event_type,
            "run_state": self._run_state,
            "process": _process_status(),
            **{key: _jsonable(value) for key, value in fields.items()},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
            handle.flush()


@dataclass(slots=True)
class FJWTimingRecorder:
    root_name: str = "fjw_workflow"
    heartbeat_writer: FJWHeartbeatWriter | None = None
    root: FJWTimingNode = field(init=False)
    _stack: list[FJWTimingNode] = field(init=False)

    def __post_init__(self) -> None:
        self.root = FJWTimingNode(name=self.root_name)
        self._stack: list[FJWTimingNode] = [self.root]

    @contextmanager
    def measure(self, name: str, **metadata: object) -> Iterator[FJWTimingNode]:
        node = FJWTimingNode(name=name, metadata=dict(metadata))
        self._stack[-1].children.append(node)
        self._stack.append(node)
        start = time.perf_counter()
        token = None
        if self.heartbeat_writer is not None:
            token = self.heartbeat_writer.start_phase(self.root_name, name, dict(metadata))
        error: str | None = None
        try:
            yield node
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            node.elapsed_seconds += time.perf_counter() - start
            self._stack.pop()
            if token is not None and self.heartbeat_writer is not None:
                self.heartbeat_writer.finish_phase(
                    token,
                    elapsed_seconds=node.elapsed_seconds,
                    error=error,
                )

    def as_jsonable(self) -> dict[str, object]:
        return self.root.as_jsonable()


@contextmanager
def maybe_measure(
    recorder: FJWTimingRecorder | None,
    name: str,
    **metadata: object,
) -> Iterator[None]:
    if recorder is None:
        yield
        return
    with recorder.measure(name, **metadata):
        yield


__all__ = [
    "FJWHeartbeatWriter",
    "FJWTimingNode",
    "FJWTimingRecorder",
    "maybe_measure",
]
