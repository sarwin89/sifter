"""SIFTER public package namespace."""

from sifter.api import AnalysisError, autofit
from sifter.config import AutofitConfig, SearchMode
from sifter.progress import ProgressCallback, ProgressEvent, ProgressPhase
from sifter.result import AnalysisSettings, FitResult, FittedPeak, ModelResult
from sifter.spectrum import Spectrum

__version__ = "0.1.0"

__all__ = [
    "AnalysisError",
    "AnalysisSettings",
    "AutofitConfig",
    "FitResult",
    "FittedPeak",
    "ModelResult",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressPhase",
    "SearchMode",
    "Spectrum",
    "__version__",
    "autofit",
]
