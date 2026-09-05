"""SIFTER public package namespace."""

from sifter.api import AnalysisError, autofit
from sifter.config import AutofitConfig, SearchMode
from sifter.context import MeasurementContext
from sifter.preview import SpectrumPreview, preview_spectrum
from sifter.progress import ProgressCallback, ProgressEvent, ProgressPhase
from sifter.reference import FitReference
from sifter.related import summarize_related_spectra
from sifter.result import AnalysisSettings, FitResult, FittedPeak, ModelResult
from sifter.spectrum import Spectrum

__version__ = "0.2.0"

__all__ = [
    "AnalysisError",
    "AnalysisSettings",
    "AutofitConfig",
    "FitResult",
    "FittedPeak",
    "FitReference",
    "MeasurementContext",
    "ModelResult",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressPhase",
    "SearchMode",
    "Spectrum",
    "SpectrumPreview",
    "__version__",
    "autofit",
    "preview_spectrum",
    "summarize_related_spectra",
]
