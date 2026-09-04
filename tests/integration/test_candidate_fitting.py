import numpy as np
import pytest

from sifter import AutofitConfig, Spectrum
from sifter.fitting import CandidateFailure, CandidateFit, fit_candidate
from sifter.models import ParameterLayout, build_candidates
from sifter.synthetic import SyntheticPeak, make_spectrum
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_single_gaussian_candidate_recovers_identifiable_truth() -> None:
    spectrum = easy_one_peak_spectrum(seed=4)
    spec = one_gaussian_spec(spectrum)

    fit = fit_candidate(spectrum, spec, starts=6, seed=9)

    assert isinstance(fit, CandidateFit)
    assert fit.peaks[0].center == pytest.approx(1.4, abs=0.01)
    assert fit.peaks[0].area == pytest.approx(2.0, rel=0.03)
    assert fit.peaks[0].sigma == pytest.approx(0.08, rel=0.05)
    assert fit.objective_rss >= 0
    assert not fit.parameters.flags.writeable
    assert not fit.components.flags.writeable


def test_weighted_fit_objective_uses_supplied_standard_deviations() -> None:
    unweighted = easy_one_peak_spectrum(seed=5)
    sigma = np.full(unweighted.x.size, 0.02)
    spectrum = Spectrum(unweighted.x, unweighted.intensity, sigma=sigma)

    fit = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=4, seed=3)

    assert isinstance(fit, CandidateFit)
    expected = float(np.sum((fit.residuals / sigma) ** 2))
    assert fit.objective_rss == pytest.approx(expected)


def test_fitted_peaks_are_canonically_sorted_by_center() -> None:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 3.0, 601),
        peaks=(
            SyntheticPeak("gaussian", area=1.0, center=0.9, sigma=0.08),
            SyntheticPeak("gaussian", area=1.4, center=2.0, sigma=0.09),
        ),
        noise="gaussian",
        snr=150.0,
        seed=10,
    )
    config = AutofitConfig(max_peaks=2, shapes=("gaussian",), baseline_orders=(0,))
    spec = next(
        item for item in build_candidates(spectrum, (), None, config) if item.peak_count == 2
    )

    fit = fit_candidate(spectrum, spec, starts=8, seed=6)

    assert isinstance(fit, CandidateFit)
    assert [peak.center for peak in fit.peaks] == sorted(peak.center for peak in fit.peaks)


def test_all_failed_starts_return_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_raises(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controlled optimizer failure")

    monkeypatch.setattr("sifter.fitting.optimizer.least_squares", always_raises)
    spectrum = easy_one_peak_spectrum()

    result = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=3, seed=2)

    assert isinstance(result, CandidateFailure)
    assert result.code == "ALL_STARTS_FAILED"
    assert result.attempted_starts == 3
    assert "controlled optimizer failure" in result.message


def test_candidate_fit_rejects_nonpositive_evaluation_budget() -> None:
    spectrum = easy_one_peak_spectrum()

    with pytest.raises(ValueError, match="max_nfev"):
        fit_candidate(
            spectrum,
            one_gaussian_spec(spectrum),
            starts=1,
            seed=2,
            max_nfev=0,
        )


def test_candidate_fit_accepts_previous_solution_as_warm_start() -> None:
    spectrum = easy_one_peak_spectrum(seed=4)
    spec = one_gaussian_spec(spectrum)
    initial = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order).initial_vector(spec)
    initial[2] *= 1.2

    fit = fit_candidate(
        spectrum,
        spec,
        starts=2,
        seed=9,
        max_nfev=2_500,
        initial_parameters=initial,
    )

    assert isinstance(fit, CandidateFit)
    assert fit.peaks[0].center == pytest.approx(1.4, abs=0.01)


def test_every_optimizer_start_receives_finite_evaluation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[object] = []

    def record_budget(*args: object, **kwargs: object) -> None:
        observed_budgets.append(kwargs.get("max_nfev"))
        raise RuntimeError("controlled optimizer stop")

    monkeypatch.setattr("sifter.fitting.optimizer.least_squares", record_budget)
    spectrum = easy_one_peak_spectrum()

    result = fit_candidate(
        spectrum,
        one_gaussian_spec(spectrum),
        starts=2,
        seed=2,
        max_nfev=37,
    )

    assert isinstance(result, CandidateFailure)
    assert observed_budgets == [37, 37]
