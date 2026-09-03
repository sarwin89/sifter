import numpy as np
import pytest

from sifter import Spectrum
from sifter.detection import detect_peak_proposals
from sifter.synthetic import SyntheticPeak, make_spectrum


def test_detector_returns_center_sorted_consensus_proposals() -> None:
    x = np.linspace(0.0, 10.0, 1001)
    spectrum, _ = make_spectrum(
        x=x,
        peaks=(
            SyntheticPeak("gaussian", area=2.0, center=3.0, sigma=0.15),
            SyntheticPeak("gaussian", area=1.5, center=7.0, sigma=0.20),
        ),
        noise="gaussian",
        snr=100.0,
        seed=2,
    )

    proposals = detect_peak_proposals(spectrum, max_peaks=6)

    assert len(proposals) == 2
    assert [proposal.center for proposal in proposals] == pytest.approx([3.0, 7.0], abs=0.08)
    assert all(proposal.width > 0 for proposal in proposals)
    assert all(proposal.prominence > 0 for proposal in proposals)
    assert all(len(proposal.sources) >= 2 for proposal in proposals)


def test_detector_deduplicates_evidence_for_one_peak() -> None:
    x = np.linspace(-1.0, 1.0, 501)
    spectrum, _ = make_spectrum(
        x=x,
        peaks=(SyntheticPeak("gaussian", area=1.0, center=0.1, sigma=0.08),),
        noise="gaussian",
        snr=80.0,
        seed=8,
    )

    proposals = detect_peak_proposals(spectrum, max_peaks=4)

    assert len(proposals) == 1
    assert proposals[0].center == pytest.approx(0.1, abs=0.03)
    assert "prominence" in proposals[0].sources


def test_detector_returns_empty_for_monotonic_signal() -> None:
    spectrum = Spectrum(np.linspace(0.0, 1.0, 101), np.linspace(1.0, 2.0, 101))

    assert detect_peak_proposals(spectrum, max_peaks=6) == ()


def test_detector_respects_max_peaks_and_is_deterministic() -> None:
    x = np.linspace(0.0, 10.0, 2001)
    peaks = tuple(
        SyntheticPeak("gaussian", area=1.0, center=center, sigma=0.08)
        for center in (1.0, 3.0, 5.0, 7.0, 9.0)
    )
    spectrum, _ = make_spectrum(x=x, peaks=peaks, noise="gaussian", snr=200.0, seed=3)

    first = detect_peak_proposals(spectrum, max_peaks=3)
    second = detect_peak_proposals(spectrum, max_peaks=3)

    assert first == second
    assert len(first) == 3


def test_detector_rejects_nonpositive_peak_limit() -> None:
    spectrum = Spectrum(np.arange(8.0), np.arange(8.0) ** 2)
    with pytest.raises(ValueError, match="max_peaks"):
        detect_peak_proposals(spectrum, max_peaks=0)
