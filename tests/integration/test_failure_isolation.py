import pytest

from sifter import AnalysisError, AutofitConfig, autofit
from sifter.fitting import CandidateFailure, CandidateFit
from sifter.fitting import fit_candidate as real_fit_candidate
from sifter.models import ModelSpec
from sifter.spectrum import Spectrum
from tests.helpers import easy_one_peak_spectrum


def test_one_failed_candidate_does_not_abort_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_only_voigt(
        spectrum: Spectrum,
        spec: ModelSpec,
        *,
        starts: int,
        seed: int,
        **kwargs: object,
    ) -> CandidateFit | CandidateFailure:
        if spec.shape == "voigt":
            return CandidateFailure(spec, "ALL_STARTS_FAILED", "controlled failure", starts)
        return real_fit_candidate(spectrum, spec, starts=starts, seed=seed, **kwargs)

    monkeypatch.setattr("sifter.search.screening.fit_candidate", fail_only_voigt)
    result = autofit(
        easy_one_peak_spectrum(),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian", "voigt"),
            baseline_orders=(0,),
            fourier=False,
        ),
    )

    assert result.best_model.shape == "gaussian"
    assert any(row.failure_code == "ALL_STARTS_FAILED" for row in result.candidates)


def test_no_valid_candidate_preserves_failures_in_analysis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(
        spectrum: Spectrum,
        spec: ModelSpec,
        *,
        starts: int,
        seed: int,
        **kwargs: object,
    ) -> CandidateFailure:
        del spectrum, seed, kwargs
        return CandidateFailure(spec, "ALL_STARTS_FAILED", "controlled failure", starts)

    monkeypatch.setattr("sifter.search.screening.fit_candidate", always_fail)

    with pytest.raises(AnalysisError) as captured:
        autofit(
            easy_one_peak_spectrum(),
            config=AutofitConfig(
                max_peaks=1,
                shapes=("gaussian",),
                baseline_orders=(0,),
                fourier=False,
            ),
        )

    assert captured.value.code == "NO_VALID_CANDIDATE"
    assert captured.value.failures
    assert {failure.code for failure in captured.value.failures} == {"ALL_STARTS_FAILED"}
