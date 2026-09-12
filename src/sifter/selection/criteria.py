"""Declared likelihood conventions, AICc/BIC scoring, and stable ranking."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from sifter.config import PeakShape
from sifter.detection import PeakProposal, detect_peak_proposals
from sifter.fitting import CandidateFailure, CandidateFit
from sifter.lineshapes import gaussian_fwhm, lorentzian_fwhm, voigt_fwhm
from sifter.models import ModelSpec, PeakStart
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    """Information criteria under one documented likelihood convention."""

    aic: float
    aicc: float | None
    bic: float


@dataclass(frozen=True, slots=True)
class ComponentDiagnostic:
    """Structural interpretation for one fitted component."""

    peak_index: int
    center: float
    maxima_spanned: int
    maxima_centers: tuple[float, ...]
    support_lower: float
    support_upper: float
    admissibility: Literal["valid", "warning", "inadmissible"]
    warning_code: str | None


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One row in the complete candidate comparison table."""

    spec: ModelSpec
    status: Literal["valid", "failed", "inadmissible"]
    parameter_count: int
    rss: float | None
    rmse: float | None
    aic: float | None
    aicc: float | None
    bic: float | None
    delta_bic: float | None
    residual_variance: float | None
    reduced_chi_squared: float | None
    warnings: tuple[str, ...]
    failure_code: str | None
    component_diagnostics: tuple[ComponentDiagnostic, ...] = ()

    @property
    def peak_count(self) -> int:
        return self.spec.peak_count

    @property
    def shape(self) -> PeakShape:
        return self.spec.shape

    @property
    def baseline_order(self) -> int:
        return self.spec.baseline_order


def unweighted_information_criteria(*, n: int, p: int, rss: float) -> InformationCriteria:
    """Score unknown-variance Gaussian residuals up to shared constants."""
    if n < 1 or p < 0 or rss < 0 or not np.isfinite(rss):
        raise ValueError("n, p, and rss must define a finite nonnegative scoring problem")
    effective_rss = max(rss, np.finfo(float).tiny)
    fit_term = n * np.log(effective_rss / n)
    aic = float(fit_term + 2 * p)
    correction = None if n <= p + 1 else float((2 * p * (p + 1)) / (n - p - 1))
    return InformationCriteria(
        aic=aic,
        aicc=None if correction is None else aic + correction,
        bic=float(fit_term + p * np.log(n)),
    )


def score_candidate(
    result: CandidateFit | CandidateFailure,
    spectrum: Spectrum,
    *,
    allow_broad_multimax_component: bool = False,
    allow_budget_exhausted: bool = False,
) -> CandidateScore:
    """Convert a fit or failure into one complete comparison row."""
    parameter_count = len(result.spec.lower_bounds)
    if isinstance(result, CandidateFailure) or (
        result.status != "converged" and not allow_budget_exhausted
    ):
        failure_code = (
            result.code if isinstance(result, CandidateFailure) else "BUDGET_EXHAUSTED"
        )
        return CandidateScore(
            spec=result.spec,
            status="failed",
            parameter_count=parameter_count,
            rss=None,
            rmse=None,
            aic=None,
            aicc=None,
            bic=None,
            delta_bic=None,
            residual_variance=None,
            reduced_chi_squared=None,
            warnings=(),
            failure_code=failure_code,
        )

    component_diagnostics = component_diagnostics_for_fit(result, spectrum)
    fatal = next(
        (
            diagnostic.warning_code
            for diagnostic in component_diagnostics
            if diagnostic.admissibility == "inadmissible"
            and diagnostic.warning_code is not None
        ),
        None,
    )
    if fatal is not None:
        return CandidateScore(
            spec=result.spec,
            status="inadmissible",
            parameter_count=parameter_count,
            rss=None,
            rmse=None,
            aic=None,
            aicc=None,
            bic=None,
            delta_bic=None,
            residual_variance=None,
            reduced_chi_squared=None,
            warnings=(fatal,),
            failure_code=fatal,
            component_diagnostics=component_diagnostics,
        )

    observation_count = spectrum.x.size
    rss = float(np.dot(result.residuals, result.residuals))
    degrees_of_freedom = observation_count - parameter_count
    if spectrum.sigma is None:
        criteria = unweighted_information_criteria(n=observation_count, p=parameter_count, rss=rss)
        reduced_chi_squared = None
    else:
        standardized = result.residuals / spectrum.sigma
        deviance = float(np.sum(np.log(2.0 * np.pi * spectrum.sigma**2) + standardized**2))
        aic = deviance + 2.0 * parameter_count
        correction = (
            None
            if observation_count <= parameter_count + 1
            else (2.0 * parameter_count * (parameter_count + 1))
            / (observation_count - parameter_count - 1)
        )
        criteria = InformationCriteria(
            aic=aic,
            aicc=None if correction is None else aic + correction,
            bic=deviance + parameter_count * np.log(observation_count),
        )
        reduced_chi_squared = (
            None
            if degrees_of_freedom <= 0
            else float(np.dot(standardized, standardized) / degrees_of_freedom)
        )
    warnings = () if criteria.aicc is not None else ("AICC_UNDEFINED",)
    if result.status == "budget_exhausted":
        warnings = (*warnings, "FAST_APPROXIMATE_FIT")
    structural_warnings = tuple(
        diagnostic.warning_code
        for diagnostic in component_diagnostics
        if diagnostic.admissibility == "warning" and diagnostic.warning_code is not None
    )
    if structural_warnings:
        warnings = (*warnings, *structural_warnings)
    if allow_broad_multimax_component and structural_warnings:
        warnings = (*warnings, "BROAD_MULTIMAX_COMPONENT_ALLOWED")
    return CandidateScore(
        spec=result.spec,
        status="valid",
        parameter_count=parameter_count,
        rss=rss,
        rmse=float(np.sqrt(rss / observation_count)),
        aic=criteria.aic,
        aicc=criteria.aicc,
        bic=criteria.bic,
        delta_bic=None,
        residual_variance=(None if degrees_of_freedom <= 0 else float(rss / degrees_of_freedom)),
        reduced_chi_squared=reduced_chi_squared,
        warnings=warnings,
        failure_code=None,
        component_diagnostics=component_diagnostics,
    )


def rank_candidates(
    scores: tuple[CandidateScore, ...], shape_order: tuple[PeakShape, ...]
) -> tuple[CandidateScore, ...]:
    """Rank valid candidates by BIC and preserve failed rows at the end."""
    if not scores:
        return ()
    family_order = {shape: index for index, shape in enumerate(shape_order)}
    eligible = [
        score
        for score in scores
        if score.status == "valid" and score.bic is not None and score.aicc is not None
    ]
    reference_bic = min((score.bic for score in eligible if score.bic is not None), default=None)
    with_deltas = tuple(
        replace(
            score,
            delta_bic=(
                None
                if score.bic is None or reference_bic is None
                else float(score.bic - reference_bic)
            ),
        )
        for score in scores
    )
    ranked = sorted(
        with_deltas,
        key=lambda score: (
            score.status != "valid" or score.aicc is None or score.bic is None,
            np.inf if score.bic is None else score.bic,
            score.parameter_count,
            score.peak_count,
            score.baseline_order,
            family_order.get(score.shape, len(family_order)),
        ),
    )
    close = [score for score in ranked if score.delta_bic is not None and score.delta_bic < 2.0]
    if len(close) >= 2:
        close_specs = {score.spec for score in close}
        ranked = [
            replace(
                score,
                warnings=score.warnings + ("AMBIGUOUS_MODEL_SELECTION",),
            )
            if score.spec in close_specs and "AMBIGUOUS_MODEL_SELECTION" not in score.warnings
            else score
            for score in ranked
        ]
    return tuple(ranked)


def component_diagnostics_for_fit(
    result: CandidateFit, spectrum: Spectrum
) -> tuple[ComponentDiagnostic, ...]:
    """Classify component support against resolved maxima without banning overlap."""
    baseline_adjusted = spectrum.intensity - result.baseline
    try:
        proposal_spectrum = Spectrum(
            spectrum.x,
            baseline_adjusted,
            sigma=spectrum.sigma,
            x_name=spectrum.x_name,
            x_unit=spectrum.x_unit,
            intensity_name=spectrum.intensity_name,
            metadata=spectrum.metadata,
        )
    except ValueError:
        return ()
    proposals = detect_peak_proposals(
        proposal_spectrum,
        max_peaks=min(max(2, result.spec.peak_count * 3), 20),
    )
    proposals = _merge_unresolved_proposals(proposals, spectrum.grid.median_step)
    diagnostics: list[ComponentDiagnostic] = []
    for index, peak in enumerate(result.peaks):
        half_width = _peak_fwhm(result.spec.shape, peak) / 2.0
        support_lower = float(peak.center - half_width)
        support_upper = float(peak.center + half_width)
        maxima_centers = tuple(
            float(proposal.center)
            for proposal in proposals
            if support_lower <= proposal.center <= support_upper
        )
        maxima_spanned = len(maxima_centers)
        if maxima_spanned > 2:
            admissibility: Literal["valid", "warning", "inadmissible"] = "inadmissible"
            warning_code = "COMPONENT_SPANS_MORE_THAN_TWO_MAXIMA"
        elif maxima_spanned == 2:
            admissibility = "warning"
            warning_code = "COMPONENT_SPANS_TWO_MAXIMA"
        else:
            admissibility = "valid"
            warning_code = None
        diagnostics.append(
            ComponentDiagnostic(
                peak_index=index,
                center=float(peak.center),
                maxima_spanned=maxima_spanned,
                maxima_centers=maxima_centers,
                support_lower=support_lower,
                support_upper=support_upper,
                admissibility=admissibility,
                warning_code=warning_code,
            )
        )
    return tuple(diagnostics)


def _merge_unresolved_proposals(
    proposals: tuple[PeakProposal, ...],
    median_step: float,
) -> tuple[PeakProposal, ...]:
    if len(proposals) < 2:
        return proposals
    ordered = sorted(proposals, key=lambda proposal: proposal.center)
    clusters: list[list[PeakProposal]] = [[ordered[0]]]
    for proposal in ordered[1:]:
        previous = clusters[-1][-1]
        resolution = max(previous.width, proposal.width, 20.0 * median_step)
        if proposal.center - previous.center <= resolution:
            clusters[-1].append(proposal)
        else:
            clusters.append([proposal])
    merged: list[PeakProposal] = []
    for cluster in clusters:
        prominence = np.asarray([proposal.prominence for proposal in cluster], dtype=np.float64)
        total_prominence = float(np.sum(prominence))
        if total_prominence > 0.0:
            center = float(
                np.average(
                    [proposal.center for proposal in cluster],
                    weights=prominence,
                )
            )
        else:
            center = float(np.mean([proposal.center for proposal in cluster]))
        merged.append(
            PeakProposal(
                center=center,
                width=max(proposal.width for proposal in cluster),
                prominence=max(proposal.prominence for proposal in cluster),
                sources=frozenset().union(*(proposal.sources for proposal in cluster)),
            )
        )
    return tuple(merged)


def _peak_fwhm(shape: str, peak: PeakStart) -> float:
    if shape == "gaussian":
        sigma = peak.sigma
        assert sigma is not None
        return gaussian_fwhm(sigma)
    if shape == "lorentzian":
        gamma = peak.gamma
        assert gamma is not None
        return lorentzian_fwhm(gamma)
    sigma = peak.sigma
    gamma = peak.gamma
    assert sigma is not None and gamma is not None
    return voigt_fwhm(sigma=sigma, gamma=gamma)
