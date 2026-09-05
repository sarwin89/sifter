from dataclasses import replace

import numpy as np

from sifter import AutofitConfig, FittedPeak, MeasurementContext, autofit
from sifter.related import summarize_related_spectra
from tests.helpers import easy_one_peak_spectrum


def test_related_summary_tracks_peak_properties_against_temperature() -> None:
    first = autofit(
        easy_one_peak_spectrum(seed=1),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
            measurement_context=MeasurementContext(temperature=295.0, temperature_unit="K"),
        ),
    )
    second = autofit(
        easy_one_peak_spectrum(seed=2),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
            measurement_context=MeasurementContext(temperature=305.0, temperature_unit="K"),
        ),
    )

    table = summarize_related_spectra((first, second), condition="temperature")

    assert list(table["condition_value"]) == [295.0, 305.0]
    assert set(table["track_id"]) == {0}
    assert {"center", "fwhm", "area", "ambiguity"} <= set(table.columns)
    assert table["ambiguity"].tolist() == ["", ""]
    assert np.all(np.isfinite(table["fwhm"]))


def test_related_summary_reports_ambiguous_peak_identity_without_forcing_track() -> None:
    first = autofit(
        easy_one_peak_spectrum(seed=3),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
            measurement_context=MeasurementContext(temperature=295.0, temperature_unit="K"),
        ),
    )
    peak = first.best_model.peaks[0]
    ambiguous = replace(
        first,
        measurement_context=MeasurementContext(temperature=305.0, temperature_unit="K"),
        best_model=replace(
            first.best_model,
            peak_count=2,
            peaks=(
                peak,
                FittedPeak(
                    area=peak.area * 0.8,
                    center=peak.center + 0.01,
                    sigma=peak.sigma,
                    gamma=peak.gamma,
                ),
            ),
        ),
    )

    table = summarize_related_spectra((first, ambiguous), condition="temperature")

    assert "ambiguous" in set(table["ambiguity"])
