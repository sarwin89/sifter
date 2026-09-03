"""SciPy least-squares adapter with candidate-level failure isolation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from sifter.fitting.multistart import generate_starts
from sifter.models import ModelSpec, ParameterLayout, PeakStart, evaluate_model
from sifter.spectrum import Spectrum

FailureCode = Literal["ALL_STARTS_FAILED", "NONFINITE_SOLUTION", "INVALID_DOF"]


@dataclass(frozen=True, slots=True)
class CandidateFit:
    """Best converged result for one candidate specification."""

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
) -> CandidateFit | CandidateFailure:
    """Fit one candidate and retain the best valid multistart result."""
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
    best_result: OptimizeResult | None = None
    best_objective = np.inf
    errors: list[str] = []

    def objective(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        residuals = evaluate_model(spectrum.x, parameters, spec).fitted - spectrum.intensity
        return residuals if spectrum.sigma is None else residuals / spectrum.sigma

    start_vectors = generate_starts(spec, count=starts, seed=seed)
    for start in start_vectors:
        try:
            result = least_squares(
                objective,
                start,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
            )
        except Exception as error:
            errors.append(str(error))
            continue
        if not result.success:
            errors.append(str(result.message))
            continue
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            errors.append("optimizer returned a non-finite solution")
            continue
        objective_rss = float(np.dot(result.fun, result.fun))
        if objective_rss < best_objective:
            best_result = result
            best_objective = objective_rss

    if best_result is None:
        code: FailureCode = (
            "NONFINITE_SOLUTION"
            if errors and all("non-finite" in message for message in errors)
            else "ALL_STARTS_FAILED"
        )
        message = errors[-1] if errors else "no optimizer start produced a result"
        return CandidateFailure(spec=spec, code=code, message=message, attempted_starts=starts)

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
