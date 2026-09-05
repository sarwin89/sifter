from dataclasses import FrozenInstanceError

import pytest

from sifter import AutofitConfig, autofit
from tests.helpers import easy_one_peak_spectrum


def test_result_records_version_provenance_and_immutable_arrays() -> None:
    spectrum = easy_one_peak_spectrum(seed=6)
    result = autofit(
        spectrum,
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
            random_seed=19,
        ),
    )

    assert result.schema_version == "sifter.fit_result.v2"
    assert result.sifter_version == "0.1.0"
    assert result.settings.random_seed == 19
    assert result.settings.search_mode == "standard"
    assert not result.settings.allow_broad_multimax_component
    assert result.to_dict()["settings"]["allow_broad_multimax_component"] is False
    assert result.observation_count == spectrum.x.size
    assert result.source_metadata == spectrum.metadata
    assert not result.x.flags.writeable
    assert not result.best_model.fitted.flags.writeable
    assert [peak.center for peak in result.best_model.peaks] == sorted(
        peak.center for peak in result.best_model.peaks
    )
    with pytest.raises(FrozenInstanceError):
        result.settings.random_seed = 20  # type: ignore[misc]


def test_override_arguments_replace_only_requested_config_values() -> None:
    spectrum = easy_one_peak_spectrum()
    base = AutofitConfig(
        max_peaks=2,
        shapes=("gaussian", "lorentzian"),
        baseline_orders=(0,),
        fourier=True,
        random_seed=2,
    )

    result = autofit(
        spectrum,
        config=base,
        max_peaks=1,
        shapes=("gaussian",),
        fourier=False,
        random_seed=9,
        search_mode="fast",
    )

    assert result.settings.max_peaks == 1
    assert result.settings.shapes == ("gaussian",)
    assert result.settings.baseline_orders == (0,)
    assert result.settings.fourier is False
    assert result.settings.random_seed == 9
    assert result.settings.search_mode == "fast"
