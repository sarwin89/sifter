# SIFTER

**Spectral Inference using Fourier Transforms for Energy Resolution**

SIFTER is a local-first scientific Python project for reproducible decomposition of one-dimensional spectra. Version 0.4 fits Gaussian, Lorentzian, and Voigt peak models in the original data domain while using Fourier-domain information only as an auxiliary diagnostic and initializer.

Fourier evidence initializes and diagnoses fits; only the original observations determine parameters and model scores. SIFTER reports uncertainty and identifiability limitations rather than claiming resolution the data cannot support.

Version 0.4 adds a lighter human-guided interface, constant-baseline-only fitting, manual peak-center hints, top-candidate visualization, and a user-facing peak measurement table with location, height, FWHM width, area, local fit quality, and plain fit notes. BIC/AICc remain available for reproducible model selection, but they are advanced diagnostics rather than the primary readout.

## Install

```bash
python -m pip install -e ".[gui]"
```

## Development

SIFTER supports Python 3.11 through 3.13. Install the development environment with:

```bash
python -m pip install -e ".[dev,gui]"
```

The v0.4 implementation extends the reviewed v0.2/v0.3 architecture with a human-readable measurement layer and Streamlit candidate explorer.

## Local GUI

Install the GUI dependencies and launch the local application:

```bash
python -m pip install -e ".[gui]"
sifter-gui
```

The browser interface runs on `localhost`; SIFTER does not upload spectra or send telemetry.

## Python API

```python
from sifter import AutofitConfig, MeasurementContext, Spectrum, autofit

spectrum = Spectrum(x, intensity, x_name="Raman shift", x_unit="cm⁻¹")
result = autofit(
    spectrum,
    config=AutofitConfig(
        max_peaks=10,
        peak_count_mode="auto",  # or "exact" to force exactly max_peaks
        shapes=("gaussian", "lorentzian", "voigt"),
        baseline_orders=(0,),
        manual_peak_centers=(),  # optional x-axis hints such as (520.1, 532.8)
        fourier=True,
        random_seed=42,
        measurement_context=MeasurementContext(temperature=300.0, temperature_unit="K"),
    ),
)

print(result.best_model.peaks)
print(result.candidates[0].delta_bic)
result.to_peak_table().to_csv("sifter.peaks.csv", index=False)
```

Warnings use stable codes and plain-language messages. Treat poor-resolution, broad-component, near-bound, correlation, truncation, and unavailable-uncertainty warnings as limits on interpretation, not optimizer noise to hide.

Read the [scientific method](docs/scientific-method.md), [result schema](docs/result-schema.md), [privacy policy](docs/privacy.md), and [benchmark guide](benchmarks/README.md).

## Privacy

Real experimental spectra must never be committed. Keep them in the ignored `data/` or `private/` directories. Public examples, tests, and benchmarks will use deterministic synthetic spectra only.
