from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import benchmarks.benchmark_candidate_runtime as runtime_module
from benchmarks.benchmark_candidate_runtime import (
    benchmark_autofit_runtime,
    benchmark_candidate_runtime,
    make_runtime_case,
)
from benchmarks.benchmark_peak_count import benchmark_peak_count
from benchmarks.benchmark_width_recovery import benchmark_width_recovery
from benchmarks.generate_resolution_grid import generate_resolution_grid


def test_resolution_grid_is_deterministic_and_covers_declared_axes() -> None:
    first = generate_resolution_grid(seeds=(1, 2))
    second = generate_resolution_grid(seeds=(1, 2))

    pd.testing.assert_frame_equal(first, second)
    assert {
        "separation_over_fwhm",
        "snr",
        "amplitude_ratio",
        "voigt_fraction",
        "samples",
        "baseline_slope",
        "seed",
    } <= set(first.columns)
    assert len(first) > 2


def test_benchmark_functions_return_tables_without_writing_by_default(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    writes: list[Path] = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, *args, **kwargs: writes.append(Path(path)),
    )

    peak_counts = benchmark_peak_count(seeds=(1,), peak_counts=(1,))
    widths = benchmark_width_recovery(seeds=(1,), shapes=("gaussian",))

    assert not writes
    assert not peak_counts.empty
    assert not widths.empty


def test_small_repeated_seed_benchmarks_report_aggregate_recovery_metrics() -> None:
    peak_counts = benchmark_peak_count(seeds=(1, 2), peak_counts=(1, 2))
    widths = benchmark_width_recovery(seeds=(1, 2), shapes=("gaussian",))

    assert {"seed", "true_peak_count", "recovered_peak_count", "correct"} <= set(
        peak_counts.columns
    )
    assert peak_counts["correct"].mean() >= 0.75
    assert {"seed", "shape", "true_fwhm", "recovered_fwhm", "relative_error"} <= set(widths.columns)
    assert widths["relative_error"].max() < 0.15


def test_runtime_cases_cover_v02_release_peak_counts() -> None:
    cases = {peak_count: make_runtime_case(peak_count) for peak_count in (1, 2, 3, 5, 8, 10)}

    for peak_count, (spectrum, spec) in cases.items():
        assert spec.peak_count == peak_count
        assert spectrum.metadata["benchmark_case"] == f"candidate-runtime-{peak_count}-peak"


def test_runtime_benchmark_reports_work_without_writing_by_default(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    writes: list[Path] = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, *args, **kwargs: writes.append(Path(path)),
    )

    table = benchmark_candidate_runtime(
        peak_counts=(3,),
        repeats=1,
        starts=1,
        max_nfev=500,
    )

    assert not writes
    assert {
        "peak_count",
        "repeat",
        "status",
        "attempted_starts",
        "converged_starts",
        "selected_evaluations",
        "total_evaluations",
        "elapsed_seconds",
    } <= set(table.columns)
    assert table.loc[0, "status"] == "converged"


def test_autofit_runtime_benchmark_passes_workers_to_pipeline(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    seen_workers: list[int] = []

    def fake_autofit(spectrum, *, config):  # type: ignore[no-untyped-def]
        seen_workers.append(config.workers)
        return SimpleNamespace(
            best_model=SimpleNamespace(
                peak_count=config.max_peaks,
                bic=12.0,
                aicc=11.0,
            ),
            candidates=(object(), object()),
            warnings=(),
        )

    monkeypatch.setattr(runtime_module, "autofit", fake_autofit)

    table = benchmark_autofit_runtime(peak_counts=(3,), repeats=1, workers=(1, 2))

    assert seen_workers == [1, 2]
    assert table["workers"].tolist() == [1, 2]
    assert set(table["status"]) == {"completed"}
    assert set(table["selected_peak_count"]) == {3}
