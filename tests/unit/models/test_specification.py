from dataclasses import asdict

import numpy as np

from sifter.models import ModelSpec, ParameterLayout, PeakStart, evaluate_model


def test_parameter_layout_counts_and_names_each_family() -> None:
    gaussian = ParameterLayout("gaussian", peak_count=2, baseline_order=1)
    lorentzian = ParameterLayout("lorentzian", peak_count=2, baseline_order=1)
    voigt = ParameterLayout("voigt", peak_count=2, baseline_order=1)

    assert gaussian.parameter_count == 8
    assert lorentzian.parameter_count == 8
    assert voigt.parameter_count == 10
    assert gaussian.names == (
        "baseline.c0",
        "baseline.c1",
        "peak.0.area",
        "peak.0.center",
        "peak.0.sigma",
        "peak.1.area",
        "peak.1.center",
        "peak.1.sigma",
    )


def test_evaluate_model_matches_hand_derived_component_sum() -> None:
    x = np.linspace(-1.0, 1.0, 101)
    spec = ModelSpec(
        shape="gaussian",
        peak_count=1,
        baseline_order=0,
        baseline_start=(0.2,),
        starts=(PeakStart(area=1.0, center=0.0, sigma=0.1),),
        lower_bounds=(-10.0, 0.0, -1.0, 0.01),
        upper_bounds=(10.0, 10.0, 1.0, 1.0),
    )
    parameters = np.array([0.2, 1.0, 0.0, 0.1])

    evaluated = evaluate_model(x, parameters, spec)

    expected_component = np.exp(-0.5 * (x / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
    assert evaluated.components.shape == (1, x.size)
    assert np.allclose(evaluated.components[0], expected_component)
    assert np.allclose(evaluated.baseline, 0.2)
    assert np.allclose(evaluated.fitted, 0.2 + expected_component)
    assert evaluated.peaks == (PeakStart(area=1.0, center=0.0, sigma=0.1),)


def test_model_spec_is_serializable_without_numpy_values() -> None:
    spec = ModelSpec(
        shape="lorentzian",
        peak_count=1,
        baseline_order=0,
        baseline_start=(0.0,),
        starts=(PeakStart(area=1.0, center=0.0, gamma=0.1),),
        lower_bounds=(-1.0, 0.0, -1.0, 0.01),
        upper_bounds=(1.0, 2.0, 1.0, 1.0),
    )

    serialized = asdict(spec)

    assert serialized["shape"] == "lorentzian"
    assert isinstance(serialized["lower_bounds"], tuple)
