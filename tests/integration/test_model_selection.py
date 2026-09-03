import numpy as np
import pytest

from sifter import AutofitConfig, Spectrum
from sifter.detection import PeakProposal
from sifter.fitting import CandidateFit, fit_candidate
from sifter.models import build_candidates
from sifter.selection import CandidateScore, rank_candidates, score_candidate
from sifter.synthetic import SyntheticPeak, make_spectrum
from tests.helpers import easy_one_peak_spectrum, easy_two_peak_spectrum


@pytest.mark.parametrize("seed", [2, 7])
def test_bic_selects_one_peak_for_easy_single_peak_spectrum(seed: int) -> None:
    spectrum = easy_one_peak_spectrum(seed=seed)
    proposals = (PeakProposal(1.4, 0.19, 1.0, frozenset({"truth-test"})),)

    winner = _fit_and_rank(spectrum, proposals, max_peaks=2)

    assert winner.peak_count == 1


@pytest.mark.parametrize("seed", [3, 11])
def test_bic_selects_two_peaks_for_resolved_spectrum(seed: int) -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=seed)
    proposals = (
        PeakProposal(1.0, 0.19, 2.0, frozenset({"truth-test"})),
        PeakProposal(1.7, 0.21, 1.5, frozenset({"truth-test"})),
    )

    winner = _fit_and_rank(spectrum, proposals, max_peaks=2)

    assert winner.peak_count == 2


def test_bic_selects_three_peaks_for_well_separated_spectrum() -> None:
    x = np.linspace(0.0, 4.0, 801)
    spectrum, _ = make_spectrum(
        x=x,
        peaks=(
            SyntheticPeak("gaussian", area=1.0, center=0.8, sigma=0.07),
            SyntheticPeak("gaussian", area=1.2, center=2.0, sigma=0.08),
            SyntheticPeak("gaussian", area=0.9, center=3.2, sigma=0.07),
        ),
        baseline=(0.1,),
        noise="gaussian",
        snr=180.0,
        seed=5,
    )
    proposals = tuple(
        PeakProposal(center, 0.18, 1.0, frozenset({"truth-test"})) for center in (0.8, 2.0, 3.2)
    )

    winner = _fit_and_rank(spectrum, proposals, max_peaks=3)

    assert winner.peak_count == 3


def _fit_and_rank(
    spectrum: Spectrum, proposals: tuple[PeakProposal, ...], *, max_peaks: int
) -> CandidateScore:
    config = AutofitConfig(
        max_peaks=max_peaks, shapes=("gaussian",), baseline_orders=(0,), random_seed=13
    )
    fits = tuple(
        fit_candidate(spectrum, spec, starts=5, seed=13 + index)
        for index, spec in enumerate(build_candidates(spectrum, proposals, None, config))
    )
    assert all(isinstance(fit, CandidateFit) for fit in fits)
    scores = tuple(score_candidate(fit, spectrum) for fit in fits)
    return rank_candidates(scores, config.shapes)[0]
