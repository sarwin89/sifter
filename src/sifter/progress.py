"""Typed parent-process progress events for analysis orchestration."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

ProgressPhase = Literal[
    "preprocessing",
    "screening",
    "expansion",
    "refinement",
    "final_fitting",
    "uncertainty",
    "completion",
]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One immutable progress update emitted outside numerical workers."""

    phase: ProgressPhase
    completed: int
    total: int
    message: str | None = None

    def __post_init__(self) -> None:
        if self.total < 0 or self.completed < 0 or self.completed > self.total:
            raise ValueError("progress requires 0 <= completed <= total")


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]
TaskProgressCallback: TypeAlias = Callable[[int, int], None]


def emit_progress(
    callback: ProgressCallback | None,
    phase: ProgressPhase,
    completed: int,
    total: int,
    *,
    message: str | None = None,
) -> None:
    if callback is not None:
        callback(ProgressEvent(phase, completed, total, message))


def progress_for_phase(
    callback: ProgressCallback | None,
    phase: ProgressPhase,
    *,
    message: str | None = None,
) -> TaskProgressCallback | None:
    if callback is None:
        return None

    def report(completed: int, total: int) -> None:
        callback(ProgressEvent(phase, completed, total, message))

    return report
