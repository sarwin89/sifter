import numpy as np

from sifter.fitting import generate_starts
from sifter.models import ParameterLayout
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_multistarts_are_seeded_in_bounds_and_include_declared_start() -> None:
    spec = one_gaussian_spec(easy_one_peak_spectrum())
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)

    first = generate_starts(spec, count=6, seed=9)
    second = generate_starts(spec, count=6, seed=9)

    assert len(first) == 6
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert np.allclose(first[0], layout.initial_vector(spec))
    lower = np.asarray(spec.lower_bounds)
    upper = np.asarray(spec.upper_bounds)
    assert all(np.all(start > lower) and np.all(start < upper) for start in first)
    assert any(not np.array_equal(first[0], start) for start in first[1:])


def test_multistarts_reject_nonpositive_count() -> None:
    spec = one_gaussian_spec(easy_one_peak_spectrum())

    try:
        generate_starts(spec, count=0, seed=9)
    except ValueError as error:
        assert "count" in str(error)
    else:
        raise AssertionError("generate_starts accepted a nonpositive count")


def test_explicit_initial_parameters_replace_declared_first_start() -> None:
    spec = one_gaussian_spec(easy_one_peak_spectrum())
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)
    initial = layout.initial_vector(spec).copy()
    initial[2] *= 1.2

    starts = generate_starts(spec, count=3, seed=9, initial_parameters=initial)

    assert np.array_equal(starts[0], initial)
    assert len(starts) == 3
    assert all(np.all(start > spec.lower_bounds) for start in starts)
    assert all(np.all(start < spec.upper_bounds) for start in starts)
