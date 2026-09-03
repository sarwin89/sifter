"""Generate the declared SIFTER resolution-benchmark design."""

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ResolutionCase:
    """One point in the synthetic resolution benchmark design."""

    separation_over_fwhm: float
    snr: float
    amplitude_ratio: float
    voigt_fraction: float
    samples: int
    baseline_slope: float
    seed: int


def generate_resolution_grid(
    *, seeds: tuple[int, ...], output_path: str | Path | None = None
) -> pd.DataFrame:
    """Return the Cartesian benchmark design without fitting or default writes."""
    axes = product(
        (0.75, 1.0, 1.5),
        (25.0, 100.0),
        (0.5, 1.0),
        (0.0, 0.5, 1.0),
        (201, 501),
        (0.0, 0.05),
        seeds,
    )
    cases = [ResolutionCase(*values) for values in axes]
    table = pd.DataFrame(asdict(case) for case in cases)
    if output_path is not None:
        table.to_csv(Path(output_path), index=False)
    return table


if __name__ == "__main__":
    raise SystemExit(
        "Import generate_resolution_grid() and pass output_path explicitly to write a table."
    )
