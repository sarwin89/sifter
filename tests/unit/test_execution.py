"""Spawn-safe candidate execution and deterministic scheduling."""

import os

import numpy as np
import pytest

from sifter.config import AutofitConfig
from sifter.execution import (
    NUMERICAL_THREAD_ENVIRONMENT,
    build_fit_tasks,
    execute_fit_tasks,
    limit_numerical_threads,
)
from sifter.fitting import CandidateFit
from sifter.models import build_candidates
from tests.helpers import easy_one_peak_spectrum


def test_parent_generates_stable_candidate_seeds() -> None:
    spectrum = easy_one_peak_spectrum()
    candidates = build_candidates(
        spectrum,
        (),
        None,
        AutofitConfig(max_peaks=1, shapes=("gaussian", "lorentzian"), baseline_orders=(0,)),
    )

    first = build_fit_tasks(spectrum, candidates, starts=2, seed=73, max_nfev=500)
    second = build_fit_tasks(spectrum, candidates, starts=2, seed=73, max_nfev=500)

    assert tuple(task.seed for task in first) == tuple(task.seed for task in second)
    assert len({task.seed for task in first}) == len(first)


def test_worker_thread_limits_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NUMERICAL_THREAD_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    limit_numerical_threads()

    assert all(os.environ[name] == "1" for name in NUMERICAL_THREAD_ENVIRONMENT)


def test_serial_and_spawn_parallel_results_are_equivalent() -> None:
    spectrum = easy_one_peak_spectrum()
    candidates = build_candidates(
        spectrum,
        (),
        None,
        AutofitConfig(max_peaks=1, shapes=("gaussian", "lorentzian"), baseline_orders=(0,)),
    )
    tasks = build_fit_tasks(spectrum, candidates, starts=2, seed=19, max_nfev=2_000)

    serial = execute_fit_tasks(tasks, workers=1)
    callback_processes: list[int] = []
    updates: list[tuple[int, int]] = []

    def record_progress(completed: int, total: int) -> None:
        callback_processes.append(os.getpid())
        updates.append((completed, total))

    parallel = execute_fit_tasks(tasks, workers=2, on_progress=record_progress)

    assert tuple(type(result) for result in parallel) == tuple(type(result) for result in serial)
    for serial_result, parallel_result in zip(serial, parallel, strict=True):
        assert isinstance(serial_result, CandidateFit)
        assert isinstance(parallel_result, CandidateFit)
        assert parallel_result.spec == serial_result.spec
        assert parallel_result.status == serial_result.status
        np.testing.assert_allclose(parallel_result.parameters, serial_result.parameters)
        np.testing.assert_allclose(parallel_result.fitted, serial_result.fitted)
        assert parallel_result.total_evaluations == serial_result.total_evaluations
    assert updates == [(1, 2), (2, 2)]
    assert set(callback_processes) == {os.getpid()}
