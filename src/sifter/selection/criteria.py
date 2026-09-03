"""Declared likelihood conventions, AICc/BIC scoring, and stable ranking."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from sifter.config import PeakShape
from sifter.fitting import CandidateFailure, CandidateFit
from sifter.models import ModelSpec
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    """Information criteria under one documented likelihood convention."""

    aic: float
    aicc: float | None
    bic: float


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One row in the complete candidate comparison table."""

    spec: ModelSpec
    status: Literal["valid", "failed"]
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


def score_candidate(result: CandidateFit | CandidateFailure, spectrum: Spectrum) -> CandidateScore:
    """Convert a fit or failure into one complete comparison row."""
    parameter_count = len(result.spec.lower_bounds)
    if isinstance(result, CandidateFailure):
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
            failure_code=result.code,
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
