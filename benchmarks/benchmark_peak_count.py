"""Small deterministic benchmark for identifiable Gaussian peak counts."""

from pathlib import Path

import numpy as np
import pandas as pd

from sifter import AutofitConfig, autofit
from sifter.synthetic import SyntheticPeak, make_spectrum


def benchmark_peak_count(
    *,
    seeds: tuple[int, ...],
    peak_counts: tuple[int, ...] = (1, 2, 3),
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit easy ground-truth cases and return per-seed recovery rows."""
    rows: list[dict[str, int | bool | float]] = []
    for seed in seeds:
        for peak_count in peak_counts:
            if peak_count not in {1, 2, 3}:
                raise ValueError("peak_counts must contain only 1, 2, or 3")
            centers = np.linspace(0.7, 3.3, peak_count)
            spectrum, _ = make_spectrum(
                x=np.linspace(0.0, 4.0, 401),
                peaks=tuple(
                    SyntheticPeak(
                        "gaussian",
                        area=1.0 + 0.15 * index,
                        center=float(center),
                        sigma=0.07,
                    )
                    for index, center in enumerate(centers)
                ),
                baseline=(0.12,),
                noise="gaussian",
                snr=180.0,
                seed=seed,
            )
            result = autofit(
                spectrum,
                config=AutofitConfig(
                    max_peaks=3,
                    shapes=("gaussian",),
                    baseline_orders=(0,),
                    fourier=False,
                    random_seed=seed,
                ),
            )
            recovered = result.best_model.peak_count
            rows.append(
                {
                    "seed": seed,
                    "true_peak_count": peak_count,
                    "recovered_peak_count": recovered,
                    "correct": recovered == peak_count,
                    "bic": result.best_model.bic,
                    "rmse": result.best_model.rmse,
                }
            )
    table = pd.DataFrame(rows)
    if output_path is not None:
        table.to_csv(Path(output_path), index=False)
    return table


if __name__ == "__main__":
    raise SystemExit(
        "Import benchmark_peak_count() and pass output_path explicitly to write results."
    )
