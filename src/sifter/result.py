"""Versioned immutable result records for SIFTER analyses."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sifter.config import (
    FitMode,
    JSONScalar,
    PeakCountMode,
    PeakShape,
    SearchMode,
    UncertaintyMode,
)
from sifter.context import MeasurementContext
from sifter.diagnostics import DiagnosticWarning, ResidualDiagnostics
from sifter.fitting import ParameterUncertainty
from sifter.fourier import FourierDiagnostics
from sifter.lineshapes.gaussian import gaussian, gaussian_fwhm
from sifter.lineshapes.lorentzian import lorentzian, lorentzian_fwhm
from sifter.lineshapes.voigt import voigt, voigt_fwhm
from sifter.reference import FitReference
from sifter.selection import CandidateScore, ComponentDiagnostic

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Complete reproducible settings used for one analysis."""

    max_peaks: int
    peak_count_mode: PeakCountMode
    shapes: tuple[PeakShape, ...]
    baseline_orders: tuple[int, ...]
    fourier: bool
    interpolate_nonuniform_fft: bool
    uncertainty: UncertaintyMode
    bootstrap_samples: int
    random_seed: int
    search_mode: SearchMode = "standard"
    workers: int = 1
    allow_broad_multimax_component: bool = False
    manual_peak_centers: tuple[float, ...] = ()
    fit_mode: FitMode = "standard"
    peak_hints: tuple[float, ...] = ()
    measurement_context: MeasurementContext | None = None
    reference: FitReference | None = None


@dataclass(frozen=True, slots=True)
class FittedPeak:
    """One canonically ordered fitted peak."""

    area: float
    center: float
    sigma: float | None
    gamma: float | None


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Public numerical result for the recommended candidate."""

    shape: PeakShape
    peak_count: int
    baseline_order: int
    parameter_names: tuple[str, ...]
    parameters: NDArray[np.float64]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    peaks: tuple[FittedPeak, ...]
    fitted: NDArray[np.float64]
    baseline: NDArray[np.float64]
    components: NDArray[np.float64]
    residuals: NDArray[np.float64]
    rss: float
    rmse: float
    aicc: float
    bic: float
    parameter_count: int
    observation_count: int
    reduced_chi_squared: float | None
    component_diagnostics: tuple[ComponentDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class FitResult:
    """Stable, versioned output of the public analysis API."""

    schema_version: Literal["sifter.fit_result.v1", "sifter.fit_result.v2", "sifter.fit_result.v3"]
    settings: AnalysisSettings
    source_metadata: Mapping[str, JSONScalar]
    x: NDArray[np.float64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64] | None
    x_name: str
    x_unit: str | None
    intensity_name: str
    best_model: ModelResult
    candidate_models: tuple[ModelResult, ...]
    candidates: tuple[CandidateScore, ...]
    fourier: FourierDiagnostics | None
    residual_diagnostics: ResidualDiagnostics
    uncertainty: ParameterUncertainty
    warnings: tuple[DiagnosticWarning, ...]
    observation_count: int
    sifter_version: str
    measurement_context: MeasurementContext | None = None
    reference: FitReference | None = None

    def plot(
        self,
        *,
        model: ModelResult | None = None,
        max_points: int | None = None,
        show_components: bool = True,
    ) -> dict[str, "go.Figure"]:
        """Build publication-neutral interactive figures from this result."""
        from sifter.plotting import plot_result

        return plot_result(
            self,
            model=model,
            max_points=max_points,
            show_components=show_components,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return one flat row per fitted peak."""
        uncertainty = _uncertainty_by_parameter(self.uncertainty)
        rows: list[dict[str, object]] = []
        for index, peak in enumerate(self.best_model.peaks):
            prefix = f"peak.{index}."
            row: dict[str, object] = {
                "shape": self.best_model.shape,
                "peak_index": index,
                "area": peak.area,
                "center": peak.center,
                "sigma": peak.sigma,
                "gamma": peak.gamma,
                "bic": self.best_model.bic,
                "aicc": self.best_model.aicc,
            }
            if index < len(self.best_model.component_diagnostics):
                diagnostic = self.best_model.component_diagnostics[index]
                row.update(
                    {
                        "maxima_spanned": diagnostic.maxima_spanned,
                        "maxima_centers": ";".join(
                            f"{center:.12g}" for center in diagnostic.maxima_centers
                        ),
                        "support_lower": diagnostic.support_lower,
                        "support_upper": diagnostic.support_upper,
                        "admissibility": diagnostic.admissibility,
                        "warning_code": diagnostic.warning_code,
                    }
                )
            for field in ("area", "center", "sigma", "gamma"):
                estimate = uncertainty.get(prefix + field)
                row[f"{field}_standard_error"] = None if estimate is None else estimate[0]
                row[f"{field}_ci_lower"] = None if estimate is None else estimate[1]
                row[f"{field}_ci_upper"] = None if estimate is None else estimate[2]
            rows.append(row)
        return pd.DataFrame(rows)

    def to_peak_table(self, *, model: ModelResult | None = None) -> pd.DataFrame:
        """Return the user-facing fitted peak measurement table."""
        selected = self.best_model if model is None else model
        rows: list[dict[str, object]] = []
        for index, peak in enumerate(selected.peaks):
            diagnostic = (
                selected.component_diagnostics[index]
                if index < len(selected.component_diagnostics)
                else None
            )
            height = _peak_height(selected.shape, peak)
            width = _peak_fwhm(selected.shape, peak)
            rows.append(
                {
                    "peak": index + 1,
                    "location": peak.center,
                    "height": height,
                    "width_fwhm": width,
                    "area": peak.area,
                    "fit_note": _fit_note(diagnostic),
                    "maxima_spanned": None if diagnostic is None else diagnostic.maxima_spanned,
                    "local_area_error": _local_area_error(
                        self, selected, index, peak, width, diagnostic
                    ),
                    "height_error": _height_error(self, selected, peak, height),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-oriented result dictionary with privacy-safe metadata."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sifter_version": self.sifter_version,
            "settings": {
                "max_peaks": self.settings.max_peaks,
                "peak_count_mode": self.settings.peak_count_mode,
                "search_mode": self.settings.search_mode,
                "shapes": self.settings.shapes,
                "baseline_orders": self.settings.baseline_orders,
                "fourier": self.settings.fourier,
                "interpolate_nonuniform_fft": self.settings.interpolate_nonuniform_fft,
                "uncertainty": self.settings.uncertainty,
                "bootstrap_samples": self.settings.bootstrap_samples,
                "random_seed": self.settings.random_seed,
                "workers": self.settings.workers,
                "allow_broad_multimax_component": (
                    self.settings.allow_broad_multimax_component
                ),
                "manual_peak_centers": self.settings.manual_peak_centers,
                "fit_mode": self.settings.fit_mode,
                "peak_hints": self.settings.peak_hints,
                "measurement_context": (
                    None
                    if self.settings.measurement_context is None
                    else self.settings.measurement_context.to_dict()
                ),
                "reference": (
                    None if self.settings.reference is None else self.settings.reference.to_dict()
                ),
            },
            "source_metadata": _safe_metadata(self.source_metadata),
            "axes": {
                "x_name": self.x_name,
                "x_unit": self.x_unit,
                "intensity_name": self.intensity_name,
                "observation_count": self.observation_count,
            },
            "x": self.x,
            "intensity": self.intensity,
            "sigma": self.sigma,
            "best_model": _model_dict(self.best_model),
            "candidate_models": [_model_dict(model) for model in self.candidate_models],
            "candidates": [_candidate_dict(score) for score in self.candidates],
            "fourier": _fourier_dict(self.fourier),
            "residual_diagnostics": {
                "mean": self.residual_diagnostics.mean,
                "standard_deviation": self.residual_diagnostics.standard_deviation,
                "durbin_watson": self.residual_diagnostics.durbin_watson,
                "lag_one_correlation": self.residual_diagnostics.lag_one_correlation,
            },
            "uncertainty": _uncertainty_dict(self.uncertainty),
            "warnings": [_warning_dict(warning) for warning in self.warnings],
        }
        if self.schema_version in {"sifter.fit_result.v2", "sifter.fit_result.v3"}:
            payload["measurement_context"] = (
                None
                if self.measurement_context is None
                else self.measurement_context.to_dict()
            )
            payload["reference"] = None if self.reference is None else self.reference.to_dict()
        return payload

    def to_json(self) -> str:
        """Serialize deterministic standards-compliant JSON with nonfinite redaction."""
        sanitized, had_nonfinite = _sanitize(self.to_dict())
        assert isinstance(sanitized, dict)
        if had_nonfinite:
            sanitized["warnings"].append(
                _warning_dict(
                    DiagnosticWarning(
                        code="NONFINITE_VALUE_OMITTED",
                        severity="warning",
                        message="one or more non-finite numeric values were encoded as null",
                        context=MappingProxyType({}),
                    )
                )
            )
        return json.dumps(sanitized, allow_nan=False, sort_keys=True)


def frozen_metadata(values: Mapping[str, JSONScalar]) -> Mapping[str, JSONScalar]:
    """Copy metadata into a read-only mapping for result assembly."""
    return MappingProxyType(dict(values))


def frozen_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy an array so result records never expose mutable state."""
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def _uncertainty_by_parameter(
    uncertainty: ParameterUncertainty,
) -> dict[str, tuple[float, float, float]]:
    if uncertainty.standard_errors is None or uncertainty.confidence_intervals is None:
        return {}
    return {
        name: (standard_error, interval[0], interval[1])
        for name, standard_error, interval in zip(
            uncertainty.parameters,
            uncertainty.standard_errors,
            uncertainty.confidence_intervals,
            strict=True,
        )
    }


def _model_dict(model: ModelResult) -> dict[str, Any]:
    return {
        "shape": model.shape,
        "peak_count": model.peak_count,
        "baseline_order": model.baseline_order,
        "parameter_names": model.parameter_names,
        "parameters": model.parameters,
        "lower_bounds": model.lower_bounds,
        "upper_bounds": model.upper_bounds,
        "peaks": [
            {
                "area": peak.area,
                "center": peak.center,
                "sigma": peak.sigma,
                "gamma": peak.gamma,
            }
            for peak in model.peaks
        ],
        "fitted": model.fitted,
        "baseline": model.baseline,
        "components": model.components,
        "residuals": model.residuals,
        "rss": model.rss,
        "rmse": model.rmse,
        "aicc": model.aicc,
        "bic": model.bic,
        "parameter_count": model.parameter_count,
        "observation_count": model.observation_count,
        "reduced_chi_squared": model.reduced_chi_squared,
        "component_diagnostics": [
            _component_diagnostic_dict(diagnostic)
            for diagnostic in model.component_diagnostics
        ],
    }


def _peak_height(shape: PeakShape, peak: FittedPeak) -> float:
    center = np.array([peak.center], dtype=np.float64)
    if shape == "gaussian":
        if peak.sigma is None:
            return float("nan")
        return float(gaussian(center, area=peak.area, center=peak.center, sigma=peak.sigma)[0])
    if shape == "lorentzian":
        if peak.gamma is None:
            return float("nan")
        return float(lorentzian(center, area=peak.area, center=peak.center, gamma=peak.gamma)[0])
    if peak.sigma is None or peak.gamma is None:
        return float("nan")
    return float(
        voigt(center, area=peak.area, center=peak.center, sigma=peak.sigma, gamma=peak.gamma)[0]
    )


def _peak_fwhm(shape: PeakShape, peak: FittedPeak) -> float:
    if shape == "gaussian":
        return float("nan") if peak.sigma is None else gaussian_fwhm(peak.sigma)
    if shape == "lorentzian":
        return float("nan") if peak.gamma is None else lorentzian_fwhm(peak.gamma)
    if peak.sigma is None or peak.gamma is None:
        return float("nan")
    return voigt_fwhm(sigma=peak.sigma, gamma=peak.gamma)


def _fit_note(diagnostic: ComponentDiagnostic | None) -> str:
    if diagnostic is None:
        return "clear"
    if diagnostic.maxima_spanned == 2:
        return "shoulder"
    if diagnostic.maxima_spanned > 2 or diagnostic.admissibility == "inadmissible":
        return "check fit"
    if diagnostic.warning_code:
        return "overlapping"
    return "clear"


def _local_area_error(
    result: FitResult,
    model: ModelResult,
    index: int,
    peak: FittedPeak,
    width: float,
    diagnostic: ComponentDiagnostic | None,
) -> float:
    if result.x.size < 2:
        return 0.0
    lower = (
        diagnostic.support_lower
        if diagnostic is not None
        else peak.center - max(width, np.finfo(float).eps)
    )
    upper = (
        diagnostic.support_upper
        if diagnostic is not None
        else peak.center + max(width, np.finfo(float).eps)
    )
    mask = (result.x >= lower) & (result.x <= upper)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(result.x - peak.center)))
        mask[nearest] = True
    residual_area = float(np.trapezoid(np.abs(model.residuals[mask]), result.x[mask]))
    component_area = float(np.trapezoid(np.abs(model.components[index][mask]), result.x[mask]))
    return residual_area / max(component_area, np.finfo(float).eps)


def _height_error(
    result: FitResult,
    model: ModelResult,
    peak: FittedPeak,
    height: float,
) -> float:
    if not np.isfinite(height) or height <= 0:
        return float("nan")
    nearest = int(np.argmin(np.abs(result.x - peak.center)))
    return float(abs(model.residuals[nearest]) / height)


def _candidate_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "shape": score.shape,
        "peak_count": score.peak_count,
        "baseline_order": score.baseline_order,
        "status": score.status,
        "parameter_count": score.parameter_count,
        "rss": score.rss,
        "rmse": score.rmse,
        "aic": score.aic,
        "aicc": score.aicc,
        "bic": score.bic,
        "delta_bic": score.delta_bic,
        "residual_variance": score.residual_variance,
        "reduced_chi_squared": score.reduced_chi_squared,
        "warnings": score.warnings,
        "failure_code": score.failure_code,
        "component_diagnostics": [
            _component_diagnostic_dict(diagnostic)
            for diagnostic in score.component_diagnostics
        ],
    }


def _component_diagnostic_dict(diagnostic: ComponentDiagnostic) -> dict[str, Any]:
    return {
        "peak_index": diagnostic.peak_index,
        "center": diagnostic.center,
        "maxima_spanned": diagnostic.maxima_spanned,
        "maxima_centers": diagnostic.maxima_centers,
        "support_lower": diagnostic.support_lower,
        "support_upper": diagnostic.support_upper,
        "admissibility": diagnostic.admissibility,
        "warning_code": diagnostic.warning_code,
    }


def _fourier_dict(fourier: FourierDiagnostics | None) -> dict[str, Any] | None:
    if fourier is None:
        return None
    return {
        "applicable": fourier.applicable,
        "interpolated": fourier.interpolated,
        "frequency": fourier.frequency,
        "magnitude": fourier.magnitude,
        "candidate_spacings": fourier.candidate_spacings,
        "window": fourier.window,
        "warning_code": fourier.warning_code,
        "envelope_fits": [
            {
                "family": fit.family,
                "intercept": fit.intercept,
                "decay_coefficients": fit.decay_coefficients,
                "rss": fit.rss,
                "bic": fit.bic,
                "frequency_min": fit.frequency_min,
                "frequency_max": fit.frequency_max,
            }
            for fit in fourier.envelope_fits
        ],
    }


def _uncertainty_dict(uncertainty: ParameterUncertainty) -> dict[str, Any]:
    return {
        "method": uncertainty.method,
        "parameters": uncertainty.parameters,
        "standard_errors": uncertainty.standard_errors,
        "confidence_intervals": uncertainty.confidence_intervals,
        "successful_bootstraps": uncertainty.successful_bootstraps,
        "requested_bootstraps": uncertainty.requested_bootstraps,
        "warning": (None if uncertainty.warning is None else _warning_dict(uncertainty.warning)),
    }


def _warning_dict(warning: DiagnosticWarning) -> dict[str, Any]:
    return {
        "code": warning.code,
        "severity": warning.severity,
        "message": warning.message,
        "context": dict(warning.context),
    }


def _safe_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, Path)) and any(
            marker in key.lower() for marker in ("path", "file")
        ):
            text = str(value)
            safe[key] = PureWindowsPath(text).name if "\\" in text else Path(text).name
        else:
            safe[key] = value
    return safe


def _sanitize(value: Any) -> tuple[Any, bool]:
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        found = False
        for key, item in value.items():
            sanitized, item_found = _sanitize(item)
            result[str(key)] = sanitized
            found = found or item_found
        return result, found
    if isinstance(value, (tuple, list)):
        result_list: list[Any] = []
        found = False
        for item in value:
            sanitized, item_found = _sanitize(item)
            result_list.append(sanitized)
            found = found or item_found
        return result_list, found
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None, True
    if isinstance(value, np.integer):
        return int(value), False
    return value, False
