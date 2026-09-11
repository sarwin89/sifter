import pytest

from sifter.config import AutofitConfig


def test_autofit_config_has_conservative_defaults() -> None:
    config = AutofitConfig()

    assert config.max_peaks == 10
    assert config.search_mode == "standard"
    assert config.peak_count_mode == "auto"
    assert config.shapes == ("gaussian", "lorentzian", "voigt")
    assert config.baseline_orders == (0,)
    assert config.fourier
    assert not config.interpolate_nonuniform_fft
    assert config.uncertainty == "covariance"
    assert config.bootstrap_samples == 250
    assert config.random_seed == 42
    assert config.workers == 1
    assert not config.allow_broad_multimax_component


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_peaks": 0}, "max_peaks"),
        ({"search_mode": "turbo"}, "search_mode"),
        ({"peak_count_mode": "range"}, "peak_count_mode"),
        ({"shapes": ()}, "shape"),
        ({"shapes": ("gaussian", "gaussian")}, "unique"),
        ({"shapes": ("pseudo_voigt",)}, "unsupported"),
        ({"baseline_orders": ()}, "baseline"),
        ({"baseline_orders": (0, 1)}, "constant"),
        ({"baseline_orders": (1,)}, "constant"),
        ({"baseline_orders": (3,)}, "baseline"),
        ({"manual_peak_centers": tuple(float(index) for index in range(10))}, "manual"),
        ({"manual_peak_centers": (1.0, 1.0)}, "unique"),
        ({"manual_peak_centers": (float("nan"),)}, "finite"),
        (
            {"peak_count_mode": "exact", "max_peaks": 1, "manual_peak_centers": (1.0, 2.0)},
            "manual",
        ),
        ({"uncertainty": "profile"}, "uncertainty"),
        ({"bootstrap_samples": 200}, "bootstrap_samples"),
        ({"random_seed": -1}, "random_seed"),
        ({"workers": 0}, "workers"),
        ({"workers": True}, "workers"),
        ({"allow_broad_multimax_component": "yes"}, "allow_broad_multimax_component"),
    ],
)
def test_autofit_config_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AutofitConfig(**kwargs)
