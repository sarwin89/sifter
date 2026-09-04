"""Deterministic serial and spawn-safe process execution for candidate fits."""

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from numpy.typing import NDArray

from sifter.config import JSONScalar
from sifter.fitting import CandidateFailure, CandidateFit, fit_candidate
from sifter.models import ModelSpec
from sifter.spectrum import Spectrum

FitFunction = Callable[..., CandidateFit | CandidateFailure]

NUMERICAL_THREAD_ENVIRONMENT = (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class SpectrumPayload:
    """Pickle-safe spectrum data transferred to spawned workers."""

    x: NDArray[np.float64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64] | None
    x_name: str
    x_unit: str | None
    intensity_name: str
    metadata: tuple[tuple[str, JSONScalar], ...]

    @classmethod
    def from_spectrum(cls, spectrum: Spectrum) -> "SpectrumPayload":
        return cls(
            x=spectrum.x,
            intensity=spectrum.intensity,
            sigma=spectrum.sigma,
            x_name=spectrum.x_name,
            x_unit=spectrum.x_unit,
            intensity_name=spectrum.intensity_name,
            metadata=tuple(spectrum.metadata.items()),
        )

    def to_spectrum(self) -> Spectrum:
        return Spectrum(
            self.x,
            self.intensity,
            sigma=self.sigma,
            x_name=self.x_name,
            x_unit=self.x_unit,
            intensity_name=self.intensity_name,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class CandidateFitTask:
    """Complete, serializable input for one candidate optimizer call."""

    spectrum: SpectrumPayload
    spec: ModelSpec
    starts: int
    seed: int
    max_nfev: int
    initial_parameters: tuple[float, ...] | None = None
    allow_budget_exhausted: bool = False


def build_fit_tasks(
    spectrum: Spectrum,
    candidates: Sequence[ModelSpec],
    *,
    starts: int,
    seed: int,
    max_nfev: int,
    initial_parameters: Mapping[ModelSpec, NDArray[np.float64]] | None = None,
    allow_budget_exhausted: bool = False,
) -> tuple[CandidateFitTask, ...]:
    """Assign stable candidate seeds in the parent before any dispatch."""
    payload = SpectrumPayload.from_spectrum(spectrum)
    seed_sequences = np.random.SeedSequence(seed).spawn(len(candidates))
    tasks: list[CandidateFitTask] = []
    for candidate, seed_sequence in zip(candidates, seed_sequences, strict=True):
        candidate_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        initial = None if initial_parameters is None else initial_parameters.get(candidate)
        tasks.append(
            CandidateFitTask(
                spectrum=payload,
                spec=candidate,
                starts=starts,
                seed=candidate_seed,
                max_nfev=max_nfev,
                initial_parameters=(
                    None if initial is None else tuple(float(value) for value in initial)
                ),
                allow_budget_exhausted=allow_budget_exhausted,
            )
        )
    return tuple(tasks)


def execute_fit_tasks(
    tasks: Sequence[CandidateFitTask],
    *,
    workers: int,
    fit_function: FitFunction = fit_candidate,
) -> tuple[CandidateFit | CandidateFailure, ...]:
    """Execute candidate tasks in input order, serially or with spawn workers."""
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1 or len(tasks) < 2:
        return tuple(_call_fit_candidate(task, fit_function) for task in tasks)
    if fit_function is not fit_candidate:
        raise ValueError("custom fit functions are supported only for serial execution")
    with _parent_thread_limit_environment(), ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=limit_numerical_threads,
    ) as executor:
        return tuple(executor.map(_fit_candidate_worker, tasks))


def limit_numerical_threads() -> None:
    """Prevent each process worker from creating nested numerical thread pools."""
    for name in NUMERICAL_THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _fit_candidate_worker(task: CandidateFitTask) -> CandidateFit | CandidateFailure:
    """Top-level spawn target containing no UI behavior."""
    limit_numerical_threads()
    return _call_fit_candidate(task, fit_candidate)


def _call_fit_candidate(
    task: CandidateFitTask,
    fit_function: FitFunction,
) -> CandidateFit | CandidateFailure:
    return fit_function(
        task.spectrum.to_spectrum(),
        task.spec,
        starts=task.starts,
        seed=task.seed,
        max_nfev=task.max_nfev,
        initial_parameters=task.initial_parameters,
        allow_budget_exhausted=task.allow_budget_exhausted,
    )


@contextmanager
def _parent_thread_limit_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in NUMERICAL_THREAD_ENVIRONMENT}
    limit_numerical_threads()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
