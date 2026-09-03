from dataclasses import replace

import numpy as np

from sifter.fitting import CandidateFit, covariance_uncertainty, fit_candidate
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_rank_deficient_jacobian_withholds_covariance() -> None:
    spectrum = easy_one_peak_spectrum()
    fit = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=3, seed=2)
    assert isinstance(fit, CandidateFit)
    rank_deficient = replace(
        fit, jacobian=np.ones((spectrum.x.size, fit.parameters.size), dtype=float)
    )

    result = covariance_uncertainty(rank_deficient, spectrum)

    assert result.parameters == ()
    assert result.standard_errors is None
    assert result.confidence_intervals is None
    assert result.warning is not None
    assert result.warning.code == "COVARIANCE_RANK_DEFICIENT"


def test_full_rank_fit_returns_finite_covariance_intervals() -> None:
    spectrum = easy_one_peak_spectrum(seed=3)
    fit = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=4, seed=4)
    assert isinstance(fit, CandidateFit)

    result = covariance_uncertainty(fit, spectrum)

    assert len(result.parameters) == fit.parameters.size
    assert result.standard_errors is not None
    assert result.confidence_intervals is not None
    assert np.isfinite(result.standard_errors).all()
    assert all(lower < upper for lower, upper in result.confidence_intervals)
    assert result.warning is None
