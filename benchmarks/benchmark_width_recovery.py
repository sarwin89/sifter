"""Deterministic single-peak width-recovery benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from sifter import AutofitConfig, autofit
from sifter.config import PeakShape
from sifter.lineshapes import gaussian_fwhm, lorentzian_fwhm, voigt_fwhm
from sifter.synthetic import SyntheticPeak, make_spectrum


def benchmark_width_recovery(
    *,
    seeds: tuple[int, ...],
    shapes: tuple[PeakShape, ...] = ("gaussian", "lorentzian", "voigt"),
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit identifiable single peaks and return relative FWHM recovery errors."""
    rows: list[dict[str, str | int | float]] = []
    for seed in seeds:
        for shape in shapes:
            peak, true_fwhm = _truth(shape)
            spectrum, _ = make_spectrum(
                x=np.linspace(-2.0, 2.0, 501),
                peaks=(peak,),
                baseline=(0.1,),
                noise="gaussian",
                snr=200.0,
                seed=seed,
            )
            result = autofit(
                spectrum,
                config=AutofitConfig(
                    max_peaks=1,
                    shapes=(shape,),
                    baseline_orders=(0,),
                    fourier=False,
                    random_seed=seed,
                ),
            )
            fitted = result.best_model.peaks[0]
            recovered_fwhm = _fitted_fwhm(shape, fitted.sigma, fitted.gamma)
            rows.append(
                {
                    "seed": seed,
                    "shape": shape,
                    "true_fwhm": true_fwhm,
                    "recovered_fwhm": recovered_fwhm,
                    "relative_error": abs(recovered_fwhm - true_fwhm) / true_fwhm,
                }
            )
    table = pd.DataFrame(rows)
    if output_path is not None:
        table.to_csv(Path(output_path), index=False)
    return table


def _truth(shape: PeakShape) -> tuple[SyntheticPeak, float]:
    if shape == "gaussian":
        return SyntheticPeak(shape, area=1.5, center=0.1, sigma=0.08), gaussian_fwhm(0.08)
    if shape == "lorentzian":
        return SyntheticPeak(shape, area=1.5, center=0.1, gamma=0.08), lorentzian_fwhm(0.08)
    return (
        SyntheticPeak(shape, area=1.5, center=0.1, sigma=0.06, gamma=0.03),
        voigt_fwhm(sigma=0.06, gamma=0.03),
    )


def _fitted_fwhm(shape: PeakShape, sigma: float | None, gamma: float | None) -> float:
    if shape == "gaussian":
        assert sigma is not None
        return gaussian_fwhm(sigma)
    if shape == "lorentzian":
        assert gamma is not None
        return lorentzian_fwhm(gamma)
    assert sigma is not None and gamma is not None
    return voigt_fwhm(sigma=sigma, gamma=gamma)


if __name__ == "__main__":
    raise SystemExit(
        "Import benchmark_width_recovery() and pass output_path explicitly to write results."
    )
