"""Repeatable candidate-level runtime cases for optimizer profiling."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sifter import AutofitConfig, Spectrum
from sifter.detection import PeakProposal
from sifter.fitting import CandidateFit, fit_candidate
from sifter.models import ModelSpec, PeakStart, build_candidates
from sifter.synthetic import SyntheticPeak, make_spectrum

SUPPORTED_RUNTIME_PEAK_COUNTS = frozenset({3, 10})


def make_runtime_case(peak_count: int) -> tuple[Spectrum, ModelSpec]:
    """Build one deterministic, well-resolved Gaussian runtime case."""
    if peak_count not in SUPPORTED_RUNTIME_PEAK_COUNTS:
        raise ValueError("runtime peak_count must be 3 or 10")

    centers = np.linspace(0.6, 9.4, peak_count)
    peaks = tuple(
        SyntheticPeak(
            "gaussian",
            area=0.9 + 0.2 * index / max(peak_count - 1, 1),
            center=float(center),
            sigma=0.08,
        )
        for index, center in enumerate(centers)
    )
    generated, _ = make_spectrum(
        x=np.linspace(0.0, 10.0, 801),
        peaks=peaks,
        baseline=(0.12,),
        noise="gaussian",
        snr=200.0,
        seed=1729 + peak_count,
    )
    spectrum = Spectrum(
        generated.x,
        generated.intensity,
        metadata={
            **dict(generated.metadata),
            "benchmark_case": f"candidate-runtime-{peak_count}-peak",
        },
    )
    proposals = tuple(
        PeakProposal(
            center=peak.center,
            width=2.354820045 * 0.08,
            prominence=peak.area,
            sources=frozenset({"synthetic_truth"}),
        )
        for peak in peaks
    )
    config = AutofitConfig(
        max_peaks=peak_count,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
        random_seed=1729,
    )
    built = build_candidates(spectrum, proposals, None, config)
    spec = next(candidate for candidate in built if candidate.peak_count == peak_count)
    return spectrum, replace(
        spec,
        baseline_start=(0.12,),
        starts=tuple(
            PeakStart(area=peak.area, center=peak.center, sigma=peak.sigma)
            for peak in peaks
        ),
    )


def benchmark_candidate_runtime(
    *,
    peak_counts: tuple[int, ...] = (3, 10),
    repeats: int = 3,
    starts: int = 2,
    max_nfev: int = 2_000,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Measure repeated candidate fits without imposing wall-clock pass/fail limits."""
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    rows: list[dict[str, object]] = []
    for peak_count in peak_counts:
        spectrum, spec = make_runtime_case(peak_count)
        for repeat in range(1, repeats + 1):
            result = fit_candidate(
                spectrum,
                spec,
                starts=starts,
                seed=1729 + repeat,
                max_nfev=max_nfev,
            )
            if isinstance(result, CandidateFit):
                status = result.status
                selected_evaluations: int | None = result.evaluations
                objective_rss: float | None = result.objective_rss
                failure_code: str | None = None
            else:
                status = "failed"
                selected_evaluations = None
                objective_rss = None
                failure_code = result.code
            rows.append(
                {
                    "peak_count": peak_count,
                    "repeat": repeat,
                    "status": status,
                    "attempted_starts": result.attempted_starts,
                    "converged_starts": result.converged_starts,
                    "selected_evaluations": selected_evaluations,
                    "total_evaluations": result.total_evaluations,
                    "elapsed_seconds": result.elapsed_seconds,
                    "objective_rss": objective_rss,
                    "failure_code": failure_code,
                }
            )

    table = pd.DataFrame(rows)
    if output_path is not None:
        table.to_csv(Path(output_path), index=False)
    return table


if __name__ == "__main__":
    raise SystemExit(
        "Import benchmark_candidate_runtime() and pass output_path explicitly to write results."
    )
