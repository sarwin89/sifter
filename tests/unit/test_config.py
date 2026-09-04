import pytest

from sifter.config import AutofitConfig


def test_autofit_config_has_conservative_defaults() -> None:
    config = AutofitConfig()

    assert config.max_peaks == 10
    assert config.search_mode == "standard"
    assert config.shapes == ("gaussian", "lorentzian", "voigt")
    assert config.baseline_orders == (0, 1, 2)
    assert config.fourier
    assert not config.interpolate_nonuniform_fft
    assert config.uncertainty == "covariance"
    assert config.bootstrap_samples == 250
    assert config.random_seed == 42


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_peaks": 0}, "max_peaks"),
        ({"search_mode": "turbo"}, "search_mode"),
        ({"shapes": ()}, "shape"),
        ({"shapes": ("gaussian", "gaussian")}, "unique"),
        ({"shapes": ("pseudo_voigt",)}, "unsupported"),
        ({"baseline_orders": ()}, "baseline"),
        ({"baseline_orders": (3,)}, "baseline"),
        ({"uncertainty": "profile"}, "uncertainty"),
        ({"bootstrap_samples": 200}, "bootstrap_samples"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_autofit_config_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AutofitConfig(**kwargs)
