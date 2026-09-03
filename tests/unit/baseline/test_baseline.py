import numpy as np
import pytest

from sifter import Spectrum
from sifter.baseline import asls_baseline, fit_polynomial_baseline


@pytest.mark.parametrize("order", [0, 1, 2])
def test_scaled_polynomial_recovers_signal_on_large_original_axis(order: int) -> None:
    x = np.linspace(1000.0, 1002.0, 201)
    z = (x - 1001.0) / 1.0
    coefficients = (3.0, -0.5, 0.25)[: order + 1]
    y = np.polynomial.polynomial.polyval(z, coefficients)
    if order == 0:
        y = y + np.linspace(-1e-9, 1e-9, x.size)

    model = fit_polynomial_baseline(Spectrum(x, y), order=order)

    assert model.x_offset == pytest.approx(1001.0)
    assert model.x_scale == pytest.approx(1.0)
    assert np.allclose(model.evaluate(x), y, atol=2e-9)


def test_asls_tracks_slow_baseline_below_a_narrow_peak() -> None:
    x = np.linspace(0.0, 1.0, 501)
    baseline = 1.0 + 0.4 * x
    peak = 3.0 * np.exp(-0.5 * ((x - 0.5) / 0.015) ** 2)

    estimated = asls_baseline(baseline + peak, smoothness=1e6, asymmetry=0.01)

    assert estimated.shape == x.shape
    assert np.isfinite(estimated).all()
    assert np.sqrt(np.mean((estimated - baseline) ** 2)) < 0.08
    assert estimated[x.size // 2] < baseline[x.size // 2] + 0.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothness": 0.0},
        {"asymmetry": 0.0},
        {"asymmetry": 1.0},
        {"iterations": 0},
    ],
)
def test_asls_rejects_invalid_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        asls_baseline(np.arange(8.0), **kwargs)
