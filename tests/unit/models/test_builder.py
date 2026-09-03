import numpy as np

from sifter import AutofitConfig, Spectrum
from sifter.detection import PeakProposal
from sifter.models import ParameterLayout, build_candidates


def test_candidate_builder_includes_every_simpler_count_deterministically() -> None:
    proposals = (
        PeakProposal(2.0, 0.2, 5.0, frozenset({"prominence"})),
        PeakProposal(5.0, 0.2, 4.0, frozenset({"derivative"})),
        PeakProposal(8.0, 0.2, 3.0, frozenset({"prominence"})),
    )
    spectrum = _example_spectrum()
    config = AutofitConfig(max_peaks=5, shapes=("gaussian",), baseline_orders=(0,))

    first = build_candidates(spectrum, proposals, None, config)
    second = build_candidates(spectrum, proposals, None, config)

    assert {spec.peak_count for spec in first} == {1, 2, 3, 4, 5}
    assert first == second
    assert all(len({peak.center for peak in spec.starts}) == spec.peak_count for spec in first)


def test_builder_crosses_shapes_and_baselines_without_mixed_families() -> None:
    config = AutofitConfig(
        max_peaks=1,
        shapes=("gaussian", "voigt"),
        baseline_orders=(0, 2),
    )

    specs = build_candidates(_example_spectrum(), (), None, config)

    assert {(spec.shape, spec.baseline_order) for spec in specs} == {
        ("gaussian", 0),
        ("gaussian", 2),
        ("voigt", 0),
        ("voigt", 2),
    }
    assert all(spec.peak_count == 1 for spec in specs)


def test_builder_uses_grid_and_span_for_positive_width_bounds() -> None:
    spectrum = _example_spectrum()
    config = AutofitConfig(max_peaks=1, shapes=("voigt",), baseline_orders=(1,))

    spec = build_candidates(spectrum, (), None, config)[0]
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)
    lower = dict(zip(layout.names, spec.lower_bounds, strict=True))
    upper = dict(zip(layout.names, spec.upper_bounds, strict=True))

    assert lower["peak.0.sigma"] >= spectrum.grid.median_step / 2.0
    assert lower["peak.0.gamma"] >= spectrum.grid.median_step / 2.0
    assert upper["peak.0.sigma"] <= spectrum.x[-1] - spectrum.x[0]
    assert upper["peak.0.gamma"] <= spectrum.x[-1] - spectrum.x[0]
    assert len(spec.lower_bounds) == len(spec.upper_bounds) == layout.parameter_count


def test_no_proposals_still_builds_one_peak_fallback() -> None:
    specs = build_candidates(
        _example_spectrum(),
        (),
        None,
        AutofitConfig(max_peaks=4, shapes=("gaussian",), baseline_orders=(0,)),
    )

    assert [spec.peak_count for spec in specs] == [1, 2, 3]
    assert all(
        spec.starts == tuple(sorted(spec.starts, key=lambda peak: peak.center)) for spec in specs
    )


def _example_spectrum() -> Spectrum:
    x = np.linspace(0.0, 10.0, 1001)
    return Spectrum(x, 0.2 + np.exp(-0.5 * ((x - 4.0) / 0.3) ** 2))
