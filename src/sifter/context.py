"""Measurement context recorded as provenance for spectral fits."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sifter.config import JSONScalar


@dataclass(frozen=True, slots=True)
class MeasurementContext:
    """Optional experimental context that never changes single-spectrum fitting."""

    temperature: float | None = None
    temperature_unit: str | None = None
    laser_power: float | None = None
    laser_power_unit: str | None = None
    conditions: Mapping[str, JSONScalar] | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            _require_temperature_unit(self.temperature_unit)
            if self.temperature_kelvin < 0.0:
                raise ValueError("temperature must be nonnegative in kelvin")
        elif self.temperature_unit is not None:
            raise ValueError("temperature_unit requires a temperature value")
        if self.laser_power is not None:
            _require_laser_power_unit(self.laser_power_unit)
            if not np.isfinite(self.laser_power) or self.laser_power < 0.0:
                raise ValueError("laser_power must be finite and nonnegative")
        elif self.laser_power_unit is not None:
            raise ValueError("laser_power_unit requires a laser_power value")
        if self.conditions is not None:
            for key, value in self.conditions.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("condition keys must be nonempty strings")
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ValueError("condition values must be JSON scalar values")

    @property
    def temperature_kelvin(self) -> float:
        """Return temperature in kelvin."""
        if self.temperature is None:
            raise ValueError("temperature is not set")
        assert self.temperature_unit is not None
        value = float(self.temperature)
        if not np.isfinite(value):
            raise ValueError("temperature must be finite")
        unit = _normalized_temperature_unit(self.temperature_unit)
        if unit == "K":
            return value
        return value + 273.15

    @property
    def laser_power_watts(self) -> float:
        """Return laser power in watts."""
        if self.laser_power is None:
            raise ValueError("laser_power is not set")
        assert self.laser_power_unit is not None
        unit = _normalized_power_unit(self.laser_power_unit)
        return float(self.laser_power) * _LASER_POWER_FACTORS[unit]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-oriented context payload."""
        payload: dict[str, object] = {}
        if self.temperature is not None:
            assert self.temperature_unit is not None
            payload["temperature"] = {
                "value": float(self.temperature),
                "unit": self.temperature_unit,
                "kelvin": self.temperature_kelvin,
            }
        if self.laser_power is not None:
            assert self.laser_power_unit is not None
            payload["laser_power"] = {
                "value": float(self.laser_power),
                "unit": self.laser_power_unit,
                "watts": self.laser_power_watts,
            }
        payload["conditions"] = dict(self.conditions or {})
        return payload


_TEMPERATURE_UNITS = {"K", "C"}
_LASER_POWER_FACTORS = {
    "nW": 1e-9,
    "uW": 1e-6,
    "microW": 1e-6,
    "mW": 1e-3,
    "W": 1.0,
}


def _require_temperature_unit(unit: str | None) -> None:
    if unit is None or _normalized_temperature_unit(unit) not in _TEMPERATURE_UNITS:
        raise ValueError("temperature_unit must be K or C")


def _require_laser_power_unit(unit: str | None) -> None:
    if unit is None or _normalized_power_unit(unit) not in _LASER_POWER_FACTORS:
        raise ValueError("laser_power_unit must be nW, uW, microW, mW, or W")


def _normalized_temperature_unit(unit: str) -> str:
    normalized = unit.strip()
    if normalized.lower() in {"k", "kelvin"}:
        return "K"
    if normalized.lower() in {"c", "celsius"}:
        return "C"
    return normalized


def _normalized_power_unit(unit: str) -> str:
    normalized = unit.strip()
    if normalized in {"uW", "microW"}:
        return normalized
    lowered = normalized.lower()
    if lowered == "nw":
        return "nW"
    if lowered == "mw":
        return "mW"
    if normalized == "W" or lowered == "w":
        return "W"
    return normalized
