import numpy as np
import pytest

from sifter import AutofitConfig, ProgressEvent, autofit
from tests.helpers import easy_one_peak_spectrum, easy_two_peak_spectrum


def test_autofit_returns_versioned_reproducible_result() -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=11)
    config = AutofitConfig(
        max_peaks=2,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=True,
        random_seed=17,
    )

    first = autofit(spectrum, config=config)
    second = autofit(spectrum, config=config)

    assert first.schema_version == "sifter.fit_result.v1"
    assert first.best_model.peak_count == 2
    assert first.settings.random_seed == 17
    assert first.best_model.rmse < 0.03
    assert first.candidates[0].delta_bic == 0.0
    assert np.array_equal(first.best_model.parameters, second.best_model.parameters)


def test_fourier_can_be_disabled_and_nonuniform_fft_fails_closed() -> None:
    uniform = easy_one_peak_spectrum()
    disabled = autofit(
        uniform,
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
        ),
    )
    nonuniform_x = uniform.x.copy()
    nonuniform_x[1:-1] += 0.001 * np.sin(np.arange(1, nonuniform_x.size - 1))
    from sifter import Spectrum

    nonuniform = Spectrum(nonuniform_x, uniform.intensity)
    unavailable = autofit(
        nonuniform,
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=True,
            interpolate_nonuniform_fft=False,
        ),
    )

    assert disabled.fourier is None
    assert unavailable.fourier is not None
    assert unavailable.fourier.applicable is False
    assert "NONUNIFORM_GRID_FFT_DISABLED" in {warning.code for warning in unavailable.warnings}


def test_config_selects_covariance_or_bootstrap_uncertainty() -> None:
    spectrum = easy_one_peak_spectrum(seed=13)
    common = dict(
        max_peaks=1,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
        random_seed=5,
    )

    covariance = autofit(spectrum, config=AutofitConfig(**common, uncertainty="covariance"))
    bootstrap = autofit(
        spectrum,
        config=AutofitConfig(
            **common,
            uncertainty="bootstrap",
            bootstrap_samples=100,
        ),
    )

    assert covariance.uncertainty.method == "covariance"
    assert bootstrap.uncertainty.method == "bootstrap"
    assert bootstrap.uncertainty.successful_bootstraps == 100


def test_standard_search_matches_exhaustive_winner_with_fewer_final_candidates() -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=21)
    common = dict(
        max_peaks=5,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
        random_seed=31,
    )

    standard = autofit(
        spectrum,
        config=AutofitConfig(**common, search_mode="standard"),
    )
    exhaustive = autofit(
        spectrum,
        config=AutofitConfig(**common, search_mode="exhaustive"),
    )

    assert standard.best_model.peak_count == exhaustive.best_model.peak_count == 2
    assert standard.best_model.bic == pytest.approx(exhaustive.best_model.bic, abs=1e-6)
    assert len(standard.candidates) < len(exhaustive.candidates)


def test_public_serial_and_spawn_parallel_searches_are_equivalent() -> None:
    spectrum = easy_one_peak_spectrum(seed=8)
    common = dict(
        max_peaks=1,
        shapes=("gaussian", "lorentzian"),
        baseline_orders=(0,),
        fourier=False,
        random_seed=23,
        search_mode="exhaustive",
    )

    serial = autofit(spectrum, config=AutofitConfig(**common, workers=1))
    parallel = autofit(spectrum, config=AutofitConfig(**common, workers=2))

    assert serial.best_model.shape == parallel.best_model.shape
    assert serial.best_model.bic == pytest.approx(parallel.best_model.bic)
    np.testing.assert_allclose(serial.best_model.parameters, parallel.best_model.parameters)
    assert parallel.settings.workers == 2


def test_progress_events_cover_standard_search_phases() -> None:
    events: list[ProgressEvent] = []

    autofit(
        easy_one_peak_spectrum(seed=9),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian", "lorentzian"),
            baseline_orders=(0,),
            fourier=False,
            random_seed=29,
            workers=2,
        ),
        progress=events.append,
    )

    phases = {event.phase for event in events}
    assert {"preprocessing", "screening", "refinement", "uncertainty", "completion"} <= phases
    assert events[0] == ProgressEvent("preprocessing", 0, 1)
    assert events[-1] == ProgressEvent("completion", 1, 1)
    assert all(event.completed <= event.total for event in events)
