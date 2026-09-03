"""Generate a deterministic two-Voigt spectrum without reading private data."""

import numpy as np

from sifter.synthetic import SyntheticPeak, make_spectrum


def main() -> None:
    spectrum, truth = make_spectrum(
        x=np.linspace(1.5, 2.0, 1001),
        peaks=(
            SyntheticPeak("voigt", area=1.0, center=1.70, sigma=0.03, gamma=0.01),
            SyntheticPeak("voigt", area=0.7, center=1.82, sigma=0.03, gamma=0.01),
        ),
        noise="gaussian",
        snr=50.0,
        seed=42,
    )
    print(
        f"Generated {spectrum.x.size} points with {len(truth.peaks)} Voigt peaks "
        f"and seed {truth.seed}."
    )


if __name__ == "__main__":
    main()
