"""Conservative covariance and residual-bootstrap uncertainty estimates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sifter.config import BOOTSTRAP_SUCCESS_FRACTION, SUPPORTED_BOOTSTRAP_SAMPLES
from sifter.fitting.optimizer import CandidateFit, fit_candidate
from sifter.models import ParameterLayout
from sifter.reporting import DiagnosticWarning, diagnostic_warning
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class ParameterUncertainty:
    """Named parameter intervals from one uncertainty method."""

    method: Literal["covariance", "bootstrap"]
    parameters: tuple[str, ...]
    standard_errors: tuple[float, ...] | None
    confidence_intervals: tuple[tuple[float, float], ...] | None
    successful_bootstraps: int | None = None
    requested_bootstraps: int | None = None
    warning: DiagnosticWarning | None = None


def covariance_uncertainty(fit: CandidateFit, spectrum: Spectrum) -> ParameterUncertainty:
    """Estimate local 95% intervals, withholding them for a singular Jacobian."""
    parameter_count = fit.parameters.size
    if np.linalg.matrix_rank(fit.jacobian) < parameter_count:
        return _withheld_covariance(
            "COVARIANCE_RANK_DEFICIENT",
            "covariance intervals were withheld because the fit Jacobian is rank deficient",
        )

    gram = fit.jacobian.T @ fit.jacobian
    try:
        covariance = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return _withheld_covariance(
            "COVARIANCE_SINGULAR",
            "covariance intervals were withheld because the information matrix is singular",
        )
    degrees_of_freedom = spectrum.x.size - parameter_count
    if degrees_of_freedom <= 0:
        return _withheld_covariance(
            "COVARIANCE_INVALID_DOF",
            "covariance intervals were withheld because residual degrees of freedom are invalid",
        )
    if spectrum.sigma is None:
        covariance *= fit.objective_rss / degrees_of_freedom
    diagonal = np.diag(covariance)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        return _withheld_covariance(
            "COVARIANCE_NONFINITE",
            "covariance intervals were withheld because finite positive variances were unavailable",
        )
    standard_errors = np.sqrt(diagonal)
    return _uncertainty_result(
        method="covariance",
        fit=fit,
        standard_errors=standard_errors,
        lower=fit.parameters - 1.96 * standard_errors,
        upper=fit.parameters + 1.96 * standard_errors,
    )


def bootstrap_uncertainty(
    fit: CandidateFit,
    spectrum: Spectrum,
    *,
    samples: int = 250,
    seed: int = 42,
) -> ParameterUncertainty:
    """Estimate parameter uncertainty by deterministic residual resampling."""
    if samples not in SUPPORTED_BOOTSTRAP_SAMPLES:
        raise ValueError("samples must be 100, 250, or 1000")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    generator = np.random.default_rng(seed)
    centered = fit.residuals - np.mean(fit.residuals)
    estimates: list[np.ndarray] = []
    for _ in range(samples):
        sampled_residuals = generator.choice(centered, size=centered.size, replace=True)
        resampled = Spectrum(
            spectrum.x,
            fit.fitted + sampled_residuals,
            sigma=spectrum.sigma,
            x_name=spectrum.x_name,
            x_unit=spectrum.x_unit,
            intensity_name=spectrum.intensity_name,
            metadata=spectrum.metadata,
        )
        refit = fit_candidate(
            resampled,
            fit.spec,
            starts=1,
            seed=int(generator.integers(0, np.iinfo(np.int32).max)),
        )
        if isinstance(refit, CandidateFit):
            estimates.append(np.asarray(refit.parameters))

    successful = len(estimates)
    minimum_successes = int(np.ceil(samples * BOOTSTRAP_SUCCESS_FRACTION))
    if successful < minimum_successes:
        return ParameterUncertainty(
            method="bootstrap",
            parameters=(),
            standard_errors=None,
            confidence_intervals=None,
            successful_bootstraps=successful,
            requested_bootstraps=samples,
            warning=diagnostic_warning(
                "BOOTSTRAP_INSUFFICIENT_SUCCESS",
                "bootstrap intervals were withheld because too few refits succeeded",
                context={"successful": successful, "requested": samples},
            ),
        )

    sample_array = np.vstack(estimates)
    standard_errors = np.std(sample_array, axis=0, ddof=1)
    lower, upper = np.percentile(sample_array, (2.5, 97.5), axis=0)
    return _uncertainty_result(
        method="bootstrap",
        fit=fit,
        standard_errors=standard_errors,
        lower=lower,
        upper=upper,
        successful_bootstraps=successful,
        requested_bootstraps=samples,
    )


def _withheld_covariance(code: str, message: str) -> ParameterUncertainty:
    return ParameterUncertainty(
        method="covariance",
        parameters=(),
        standard_errors=None,
        confidence_intervals=None,
        warning=diagnostic_warning(code, message),
    )


def _uncertainty_result(
    *,
    method: Literal["covariance", "bootstrap"],
    fit: CandidateFit,
    standard_errors: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    successful_bootstraps: int | None = None,
    requested_bootstraps: int | None = None,
) -> ParameterUncertainty:
    layout = ParameterLayout(fit.spec.shape, fit.spec.peak_count, fit.spec.baseline_order)
    return ParameterUncertainty(
        method=method,
        parameters=layout.names,
        standard_errors=tuple(float(value) for value in standard_errors),
        confidence_intervals=tuple(
            (float(low), float(high)) for low, high in zip(lower, upper, strict=True)
        ),
        successful_bootstraps=successful_bootstraps,
        requested_bootstraps=requested_bootstraps,
    )
