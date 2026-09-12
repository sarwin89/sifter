import numpy as np
import pytest

from sifter import FitConfig, fit_spectrum
from sifter.synthetic import SyntheticPeak, make_spectrum


def test_fit_config_limits_interactive_peak_count() -> None:
    with pytest.raises(ValueError, match="max_peaks cannot exceed 10"):
        FitConfig(max_peaks=11)


def test_fit_spectrum_returns_v3_result_with_new_settings() -> None:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 3.0, 241),
        peaks=(SyntheticPeak("gaussian", area=2.0, center=1.2, sigma=0.08),),
        baseline=(0.2,),
        noise="gaussian",
        snr=200.0,
        seed=12,
    )

    result = fit_spectrum(
        spectrum,
        config=FitConfig(
            max_peaks=3,
            count_mode="auto",
            fit_mode="fast",
            shapes=("gaussian",),
            peak_hints=(1.2,),
        ),
    )

    assert result.schema_version == "sifter.fit_result.v3"
    assert result.settings.fit_mode == "fast"
    assert result.settings.peak_hints == (1.2,)
    assert result.settings.peak_count_mode == "auto"
    assert result.to_dict()["settings"]["fit_mode"] == "fast"
    assert result.to_peak_table().loc[0, "location"] == pytest.approx(1.2, abs=0.05)
