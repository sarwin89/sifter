"""SciPy least-squares adapter with candidate-level failure isolation."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares

from sifter.fitting.multistart import generate_starts
from sifter.models import ModelSpec, ParameterLayout, PeakStart, evaluate_model
from sifter.spectrum import Spectrum

FailureCode = Literal["ALL_STARTS_FAILED", "NONFINITE_SOLUTION", "INVALID_DOF"]
FitStatus = Literal["converged", "budget_exhausted"]


@dataclass(frozen=True, slots=True)
class CandidateFit:
    """Best admissible result for one candidate specification."""

    spec: ModelSpec
    parameters: NDArray[np.float64]
    peaks: tuple[PeakStart, ...]
    baseline: NDArray[np.float64]
    components: NDArray[np.float64]
    fitted: NDArray[np.float64]
    residuals: NDArray[np.float64]
    objective_rss: float
    jacobian: NDArray[np.float64]
    optimality: float
    evaluations: int
    status: FitStatus = "converged"
    attempted_starts: int = 1
    converged_starts: int = 1
    total_evaluations: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    """A failed candidate row that remains visible to callers."""

    spec: ModelSpec
    code: FailureCode
    message: str
    attempted_starts: int


def fit_candidate(
    spectrum: Spectrum,
    spec: ModelSpec,
    *,
    starts: int = 8,
    seed: int = 42,
    max_nfev: int = 10_000,
    initial_parameters: ArrayLike | None = None,
    allow_budget_exhausted: bool = False,
) -> CandidateFit | CandidateFailure:
    """Fit one candidate and retain the best admissible multistart result.

    Budget-exhausted results are failures unless screening code explicitly opts in.
    Even when enabled, a converged start always outranks a provisional one.
    """
    if isinstance(max_nfev, bool) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)
    if spectrum.x.size <= layout.parameter_count:
        return CandidateFailure(
            spec=spec,
            code="INVALID_DOF",
            message="candidate has no residual degrees of freedom",
            attempted_starts=0,
        )

    lower = np.asarray(spec.lower_bounds, dtype=np.float64)
    upper = np.asarray(spec.upper_bounds, dtype=np.float64)
    best_converged: OptimizeResult | None = None
    best_converged_objective = np.inf
    best_provisional: OptimizeResult | None = None
    best_provisional_objective = np.inf
    errors: list[str] = []
    attempted_starts = 0
    converged_starts = 0
    total_evaluations = 0
    started_at = perf_counter()

    def objective(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        residuals = evaluate_model(spectrum.x, parameters, spec).fitted - spectrum.intensity
        return residuals if spectrum.sigma is None else residuals / spectrum.sigma

    start_vectors = generate_starts(
        spec,
        count=starts,
        seed=seed,
        initial_parameters=initial_parameters,
    )
    for start in start_vectors:
        attempted_starts += 1
        try:
            result = least_squares(
                objective,
                start,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except Exception as error:
            errors.append(str(error))
            continue
        total_evaluations += int(result.nfev)
        is_budget_exhausted = int(result.status) == 0
        if not result.success and not (allow_budget_exhausted and is_budget_exhausted):
            errors.append(str(result.message))
            continue
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            errors.append("optimizer returned a non-finite solution")
            continue
        objective_rss = float(np.dot(result.fun, result.fun))
        if result.success:
            converged_starts += 1
            if objective_rss < best_converged_objective:
                best_converged = result
                best_converged_objective = objective_rss
        elif objective_rss < best_provisional_objective:
            best_provisional = result
            best_provisional_objective = objective_rss

    best_result = best_converged if best_converged is not None else best_provisional
    status: FitStatus = "converged" if best_converged is not None else "budget_exhausted"

    if best_result is None:
        code: FailureCode = (
            "NONFINITE_SOLUTION"
            if errors and all("non-finite" in message for message in errors)
            else "ALL_STARTS_FAILED"
        )
        message = errors[-1] if errors else "no optimizer start produced a result"
        return CandidateFailure(
            spec=spec,
            code=code,
            message=message,
            attempted_starts=attempted_starts,
        )

    parameters = np.asarray(best_result.x, dtype=np.float64)
    jacobian = np.asarray(best_result.jac, dtype=np.float64)
    parameters, jacobian = _canonicalize(parameters, jacobian, layout)
    evaluated = evaluate_model(spectrum.x, parameters, spec)
    residuals = evaluated.fitted - spectrum.intensity
    weighted = residuals if spectrum.sigma is None else residuals / spectrum.sigma
    return CandidateFit(
        spec=spec,
        parameters=_frozen(parameters),
        peaks=evaluated.peaks,
        baseline=_frozen(evaluated.baseline),
        components=_frozen(evaluated.components),
        fitted=_frozen(evaluated.fitted),
        residuals=_frozen(residuals),
        objective_rss=float(np.dot(weighted, weighted)),
        jacobian=_frozen(jacobian),
        optimality=float(best_result.optimality),
        evaluations=int(best_result.nfev),
        status=status,
        attempted_starts=attempted_starts,
        converged_starts=converged_starts,
        total_evaluations=total_evaluations,
        elapsed_seconds=perf_counter() - started_at,
    )


def _canonicalize(
    parameters: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    layout: ParameterLayout,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    peaks = layout.decode_peaks(parameters)
    order = np.argsort([peak.center for peak in peaks])
    baseline_count = layout.baseline_order + 1
    block_size = 4 if layout.shape == "voigt" else 3
    permutation = list(range(baseline_count))
    for peak_index in order:
        start = baseline_count + int(peak_index) * block_size
        permutation.extend(range(start, start + block_size))
    return parameters[permutation], jacobian[:, permutation]


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
