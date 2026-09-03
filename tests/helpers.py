"""Reusable deterministic builders; no fixture reads files or private data."""

import numpy as np

from sifter.config import AutofitConfig, PeakShape
from sifter.models import ModelSpec, build_candidates
from sifter.spectrum import Spectrum
from sifter.synthetic import SyntheticPeak, SyntheticTruth, make_spectrum


def single_peak_spectrum(*, shape: PeakShape, points: int = 4097) -> Spectrum:
    widths: dict[PeakShape, dict[str, float]] = {
        "gaussian": {"sigma": 0.08},
        "lorentzian": {"gamma": 0.08},
        "voigt": {"sigma": 0.06, "gamma": 0.03},
    }
    peak = SyntheticPeak(shape, area=1.0, center=0.0, **widths[shape])
    spectrum, _ = make_spectrum(x=np.linspace(-2.0, 2.0, points), peaks=(peak,))
    return spectrum


def example_spectrum() -> Spectrum:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 10.0, 1001),
        peaks=(SyntheticPeak("gaussian", area=2.0, center=4.0, sigma=0.25),),
        noise="gaussian",
        snr=50.0,
        seed=42,
    )
    return spectrum


def easy_one_peak_spectrum(*, seed: int = 42) -> Spectrum:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 3.0, 601),
        peaks=(SyntheticPeak("gaussian", area=2.0, center=1.4, sigma=0.08),),
        baseline=(0.2,),
        noise="gaussian",
        snr=200.0,
        seed=seed,
    )
    return spectrum


def easy_two_peak_spectrum(*, seed: int = 42) -> tuple[Spectrum, SyntheticTruth]:
    return make_spectrum(
        x=np.linspace(0.0, 3.0, 601),
        peaks=(
            SyntheticPeak("gaussian", area=2.0, center=1.0, sigma=0.08),
            SyntheticPeak("gaussian", area=1.5, center=1.7, sigma=0.09),
        ),
        baseline=(0.2,),
        noise="gaussian",
        snr=150.0,
        seed=seed,
    )


def one_gaussian_spec(spectrum: Spectrum) -> ModelSpec:
    config = AutofitConfig(max_peaks=1, shapes=("gaussian",), baseline_orders=(0,))
    return build_candidates(spectrum, (), None, config)[0]
