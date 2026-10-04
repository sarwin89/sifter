"""SIFTER public package namespace."""

from sifter.api import AnalysisError, autofit, fit_spectrum
from sifter.config import AutofitConfig, CountMode, FitConfig, FitMode, PeakCountMode, SearchMode
from sifter.context import MeasurementContext
from sifter.preview import SpectrumPreview, preview_spectrum
from sifter.progress import ProgressCallback, ProgressEvent, ProgressPhase
from sifter.reference import FitReference
from sifter.related import summarize_related_spectra
from sifter.result import AnalysisSettings, FitResult, FittedPeak, ModelResult
from sifter.spectrum import Spectrum

__version__ = "0.4.2"

__all__ = [
    "AnalysisError",
    "AnalysisSettings",
    "AutofitConfig",
    "CountMode",
    "FitResult",
    "FitConfig",
    "FitMode",
    "FittedPeak",
    "FitReference",
    "MeasurementContext",
    "ModelResult",
    "PeakCountMode",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressPhase",
    "SearchMode",
    "Spectrum",
    "SpectrumPreview",
    "__version__",
    "autofit",
    "fit_spectrum",
    "preview_spectrum",
    "summarize_related_spectra",
]
