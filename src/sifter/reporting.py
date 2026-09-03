"""Shared structured messages emitted by scientific analyses."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sifter.config import JSONValue


@dataclass(frozen=True, slots=True)
class DiagnosticWarning:
    """Machine-readable warning with human-readable context."""

    code: str
    severity: str
    message: str
    context: Mapping[str, JSONValue]


def diagnostic_warning(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    context: Mapping[str, JSONValue] | None = None,
) -> DiagnosticWarning:
    """Build a warning whose context cannot be mutated by callers."""
    return DiagnosticWarning(
        code=code,
        severity=severity,
        message=message,
        context=MappingProxyType(dict(context or {})),
    )
