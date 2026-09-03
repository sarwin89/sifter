from dataclasses import replace

import numpy as np
import pytest

from sifter import AutofitConfig, Spectrum
from sifter.fitting import CandidateFailure, CandidateFit
from sifter.models import build_candidates
from sifter.selection import (
    rank_candidates,
    score_candidate,
    unweighted_information_criteria,
)
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_unweighted_information_criteria_match_declared_formula() -> None:
    score = unweighted_information_criteria(n=100, p=4, rss=25.0)
    aic = 100 * np.log(25.0 / 100.0) + 8.0

    assert score.aic == pytest.approx(aic)
    assert score.aicc == pytest.approx(aic + (2.0 * 4.0 * 5.0) / 95.0)
    assert score.bic == pytest.approx(100 * np.log(0.25) + 4 * np.log(100))


def test_aicc_is_undefined_without_required_sample_margin() -> None:
    score = unweighted_information_criteria(n=5, p=4, rss=2.0)

    assert score.aicc is None
    assert np.isfinite(score.aic)
    assert np.isfinite(score.bic)


def test_weighted_score_uses_known_sigma_likelihood_and_reduced_chi_squared() -> None:
    base = easy_one_peak_spectrum()
    spectrum = Spectrum(base.x, base.intensity, sigma=np.full(base.x.size, 0.2))
    fit = _candidate_fit(spectrum, peak_count=1, residual_value=0.1)

    score = score_candidate(fit, spectrum)

    expected_deviance = spectrum.x.size * np.log(2.0 * np.pi * 0.2**2) + np.sum(
        (fit.residuals / 0.2) ** 2
    )
    parameter_count = fit.parameters.size
    assert score.aic == pytest.approx(expected_deviance + 2.0 * parameter_count)
    assert score.reduced_chi_squared == pytest.approx(
        np.sum((fit.residuals / 0.2) ** 2) / (spectrum.x.size - parameter_count)
    )


def test_tie_breaking_prefers_simpler_candidate_and_warns() -> None:
    spectrum = easy_one_peak_spectrum()
    simple = _candidate_fit(spectrum, peak_count=1, residual_value=0.1)
    complex_fit = _candidate_fit(spectrum, peak_count=2, residual_value=0.1)
    complex_score = score_candidate(complex_fit, spectrum)
    simple_score = score_candidate(simple, spectrum)
    adjusted_complex = replace(complex_score, bic=simple_score.bic)

    ranked = rank_candidates((adjusted_complex, simple_score), ("gaussian", "lorentzian", "voigt"))

    assert ranked[0].peak_count == 1
    assert ranked[0].delta_bic == 0.0
    assert "AMBIGUOUS_MODEL_SELECTION" in ranked[0].warnings


def test_failed_candidate_remains_a_visible_unranked_row() -> None:
    spectrum = easy_one_peak_spectrum()
    spec = one_gaussian_spec(spectrum)
    failure = CandidateFailure(spec, "ALL_STARTS_FAILED", "controlled", attempted_starts=3)

    row = score_candidate(failure, spectrum)

    assert row.status == "failed"
    assert row.bic is None
    assert row.failure_code == "ALL_STARTS_FAILED"


def _candidate_fit(spectrum: Spectrum, *, peak_count: int, residual_value: float) -> CandidateFit:
    config = AutofitConfig(max_peaks=peak_count, shapes=("gaussian",), baseline_orders=(0,))
    spec = next(
        candidate
        for candidate in build_candidates(spectrum, (), None, config)
        if candidate.peak_count == peak_count
    )
    parameters = np.zeros(len(spec.lower_bounds), dtype=float)
    residuals = np.full(spectrum.x.size, residual_value)
    return CandidateFit(
        spec=spec,
        parameters=parameters,
        peaks=spec.starts,
        baseline=np.zeros_like(spectrum.x),
        components=np.zeros((peak_count, spectrum.x.size)),
        fitted=spectrum.intensity + residuals,
        residuals=residuals,
        objective_rss=float(np.sum(residuals**2)),
        jacobian=np.ones((spectrum.x.size, parameters.size)),
        optimality=0.0,
        evaluations=1,
    )
