# SIFTER

**Spectral Inference using Fourier Transforms for Energy Resolution**

SIFTER is a local-first scientific Python project for reproducible decomposition of one-dimensional spectra. Version 0.4.2 fits Gaussian, Lorentzian, and Voigt peak models in the original data domain while using Fourier-domain information only as an auxiliary diagnostic and initializer.

Fourier evidence initializes and diagnoses fits; only the original observations determine parameters and model scores. SIFTER reports uncertainty and identifiability limitations rather than claiming resolution the data cannot support.

Version 0.4.2 rebuilds the local app as a fast peak workbench: constant-baseline-only fitting, detected maxima review, click-to-add peak hints through Streamlit custom components v2, lightweight charts, quality-first top-candidate browsing, and a user-facing peak measurement table with location, height, FWHM width, area, local fit quality, and plain fit notes. BIC/AICc remain available for reproducibility, but they are advanced diagnostics rather than the primary readout.

## Install

```bash
python -m pip install -e ".[gui]"
```

## Development

SIFTER supports Python 3.11 through 3.13. Install the development environment with:

```bash
python -m pip install -e ".[dev,gui]"
```

The v0.4.2 implementation extends the reviewed v0.2/v0.3 architecture with a human-readable measurement layer, faster screened fitting, and a Streamlit workbench for human-guided peak review.

## Local GUI

Install the GUI dependencies and launch the local application:

```bash
python -m pip install -e ".[gui]"
sifter-gui
```

The browser interface runs on `localhost`; SIFTER does not upload spectra or send telemetry.

## Python API

```python
from sifter import FitConfig, MeasurementContext, Spectrum, fit_spectrum

spectrum = Spectrum(x, intensity, x_name="Raman shift", x_unit="cm⁻¹")
result = fit_spectrum(
    spectrum,
    config=FitConfig(
        max_peaks=10,  # interactive workbench ceiling
        count_mode="auto",  # or "exact" to force exactly max_peaks
        fit_mode="fast",  # standard/thorough/research spend more time
        shapes=("gaussian", "lorentzian", "voigt"),
        peak_hints=(),  # optional x-axis hints such as (520.1, 532.8)
        fourier=True,
        random_seed=42,
        measurement_context=MeasurementContext(temperature=300.0, temperature_unit="K"),
    ),
)

print(result.best_model.peaks)
result.to_peak_table().to_csv("sifter.peaks.csv", index=False)
```

`autofit(...)` remains available as a compatibility wrapper for the older statistical API. New UI and docs use `fit_spectrum(...)`, which returns the v3 result schema and quality-ordered retained candidates. Warnings use stable codes and plain-language messages. Treat fast-approximate, poor-resolution, broad-component, near-bound, correlation, truncation, and unavailable-uncertainty warnings as limits on interpretation, not optimizer noise to hide.

Read the [scientific method](docs/scientific-method.md), [result schema](docs/result-schema.md), [privacy policy](docs/privacy.md), and [benchmark guide](benchmarks/README.md).

## Privacy

Real experimental spectra must never be committed. Keep them in the ignored `data/` or `private/` directories. Public examples, tests, and benchmarks will use deterministic synthetic spectra only.
