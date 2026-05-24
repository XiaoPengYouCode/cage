from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


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
class FJWTimingRecorder:
    root_name: str = "fjw_workflow"
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
        try:
            yield node
        finally:
            node.elapsed_seconds += time.perf_counter() - start
            self._stack.pop()

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
    "FJWTimingNode",
    "FJWTimingRecorder",
    "maybe_measure",
]
