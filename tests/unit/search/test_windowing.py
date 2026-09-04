import numpy as np
import pytest

from sifter import AutofitConfig, Spectrum
from sifter.baseline import fit_polynomial_baseline
from sifter.detection import PeakProposal
from sifter.models import ModelSpec
from sifter.search import PeakDetectionSummary, SearchPreprocessing, search_policy
from sifter.search.windowing import build_windowed_candidates, plan_peak_windows
from sifter.synthetic import SyntheticPeak, make_spectrum


def test_window_planner_clips_boundary_peak_and_uses_lorentzian_halo_minimum() -> None:
    spectrum = Spectrum(
        np.linspace(0.0, 100.0, 101),
        np.linspace(0.0, 1.0, 101),
    )
    preprocessing = _preprocessing(
        spectrum,
        (
            _proposal(center=1.0, width=2.0),
            _proposal(center=90.0, width=3.0),
        ),
    )

    windows = plan_peak_windows(
        spectrum,
        preprocessing,
        AutofitConfig(max_peaks=2, shapes=("voigt",), baseline_orders=(0,)),
    )

    assert len(windows) == 2
    assert windows[0].fit_start_index == 0
    assert windows[0].core_start <= 1.0 <= windows[0].core_stop
    assert windows[0].fit_stop - windows[0].core_stop >= 10.0 - 1e-12


def test_window_planner_merges_excessively_overlapping_windows() -> None:
    spectrum = Spectrum(np.linspace(0.0, 100.0, 201), np.linspace(0.0, 1.0, 201))
    preprocessing = _preprocessing(
        spectrum,
        (
            _proposal(center=50.0, width=6.0, prominence=4.0),
            _proposal(center=56.0, width=6.0, prominence=3.5),
        ),
    )

    windows = plan_peak_windows(
        spectrum,
        preprocessing,
        AutofitConfig(max_peaks=2, shapes=("gaussian",), baseline_orders=(0,)),
    )

    assert len(windows) == 1
    assert tuple(maximum.center for maximum in windows[0].maxima) == (50.0, 56.0)
    assert windows[0].core_start <= 50.0
    assert windows[0].core_stop >= 56.0


def test_windowed_candidates_use_global_baseline_and_return_only_global_specs() -> None:
    x = np.linspace(-5.0, 5.0, 301)
    spectrum, _ = make_spectrum(
        x=x,
        peaks=(
            SyntheticPeak("gaussian", area=1.4, center=-2.0, sigma=0.18),
            SyntheticPeak("gaussian", area=1.0, center=2.1, sigma=0.20),
        ),
        baseline=(0.4, 0.08),
        noise="gaussian",
        snr=100.0,
        seed=12,
    )
    preprocessing = _preprocessing(
        spectrum,
        (
            _proposal(center=-2.0, width=0.42, prominence=1.4),
            _proposal(center=2.1, width=0.47, prominence=1.0),
        ),
    )
    config = AutofitConfig(
        max_peaks=3,
        shapes=("gaussian",),
        baseline_orders=(1,),
        fourier=False,
        random_seed=12,
    )

    candidates = build_windowed_candidates(
        spectrum,
        preprocessing,
        config,
        search_policy("standard"),
        seed=12,
        workers=1,
    )

    expected_baseline = fit_polynomial_baseline(spectrum, order=1).coefficients
    assert candidates
    assert all(isinstance(candidate, ModelSpec) for candidate in candidates)
    assert {candidate.peak_count for candidate in candidates} <= {2, 3}
    for candidate in candidates:
        assert candidate.baseline_start == pytest.approx(expected_baseline)
        assert all(spectrum.x[0] <= peak.center <= spectrum.x[-1] for peak in candidate.starts)


def test_windowed_candidates_add_residual_second_wave_without_exporting_local_scores() -> None:
    x = np.linspace(-5.0, 5.0, 301)
    spectrum, _ = make_spectrum(
        x=x,
        peaks=(
            SyntheticPeak("gaussian", area=1.7, center=-2.0, sigma=0.18),
            SyntheticPeak("gaussian", area=0.9, center=2.0, sigma=0.18),
        ),
        baseline=(0.2,),
        seed=9,
    )
    preprocessing = _preprocessing(
        spectrum,
        (_proposal(center=-2.0, width=0.42, prominence=1.7),),
    )
    config = AutofitConfig(
        max_peaks=2,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
        random_seed=9,
    )

    candidates = build_windowed_candidates(
        spectrum,
        preprocessing,
        config,
        search_policy("standard"),
        seed=9,
        workers=1,
    )

    assert any(candidate.peak_count == 2 for candidate in candidates)
    assert not any(hasattr(candidate, "screening_bic") for candidate in candidates)


def _proposal(
    *,
    center: float,
    width: float,
    prominence: float = 1.0,
) -> PeakProposal:
    return PeakProposal(
        center=center,
        width=width,
        prominence=prominence,
        sources=frozenset({"test"}),
    )


def _preprocessing(
    spectrum: Spectrum,
    proposals: tuple[PeakProposal, ...],
) -> SearchPreprocessing:
    return SearchPreprocessing(
        baseline=np.zeros_like(spectrum.x),
        adjusted=spectrum.intensity,
        proposal_spectrum=spectrum,
        proposals=proposals,
        detection=PeakDetectionSummary(
            detected_count=len(proposals),
            centers=tuple(proposal.center for proposal in proposals),
            median_width=float(np.median([proposal.width for proposal in proposals]))
            if proposals
            else None,
            strongest_prominence=max((proposal.prominence for proposal in proposals), default=None),
        ),
        fourier=None,
    )
