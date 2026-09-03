import numpy as np

from sifter.fitting import CandidateFit, bootstrap_uncertainty, fit_candidate
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_residual_bootstrap_is_deterministic_for_fixed_seed() -> None:
    spectrum = easy_one_peak_spectrum(seed=8)
    fit = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=4, seed=3)
    assert isinstance(fit, CandidateFit)

    first = bootstrap_uncertainty(fit, spectrum, samples=100, seed=12)
    second = bootstrap_uncertainty(fit, spectrum, samples=100, seed=12)

    assert first.parameters == second.parameters
    assert first.standard_errors == second.standard_errors
    assert first.confidence_intervals == second.confidence_intervals
    assert first.successful_bootstraps == second.successful_bootstraps == 100
    assert first.warning is None
    assert first.standard_errors is not None
    assert np.isfinite(first.standard_errors).all()
