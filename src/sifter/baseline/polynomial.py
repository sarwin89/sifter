"""Numerically conditioned polynomial baseline models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class ScaledPolynomial:
    """Polynomial coefficients evaluated on a centered, scaled coordinate."""

    coefficients: tuple[float, ...]
    x_offset: float
    x_scale: float

    def evaluate(self, x: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(x, dtype=np.float64)
        scaled = (values - self.x_offset) / self.x_scale
        return np.asarray(
            np.polynomial.polynomial.polyval(scaled, self.coefficients),
            dtype=np.float64,
        )


def fit_polynomial_baseline(spectrum: Spectrum, *, order: int) -> ScaledPolynomial:
    """Fit a polynomial of order zero, one, or two to a spectrum."""
    if order not in {0, 1, 2}:
        raise ValueError("baseline order must be 0, 1, or 2")
    x_offset = float((spectrum.x[0] + spectrum.x[-1]) / 2.0)
    x_scale = float((spectrum.x[-1] - spectrum.x[0]) / 2.0)
    scaled = (spectrum.x - x_offset) / x_scale
    coefficients = np.polynomial.polynomial.polyfit(scaled, spectrum.intensity, order)
    return ScaledPolynomial(
        coefficients=tuple(float(value) for value in coefficients),
        x_offset=x_offset,
        x_scale=x_scale,
    )
