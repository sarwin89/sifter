# SIFTER

**Spectral Inference using Fourier Transforms for Energy Resolution**

SIFTER is a local-first scientific Python project for reproducible decomposition of one-dimensional spectra. Version 0.3 fits Gaussian, Lorentzian, and Voigt peak models in the original data domain while using Fourier-domain information only as an auxiliary diagnostic and initializer.

Fourier evidence initializes and diagnoses fits; only the original observations determine parameters and model scores. SIFTER reports uncertainty and identifiability limitations rather than claiming resolution the data cannot support.

Version 0.3 adds explicit auto-vs-exact peak-count fitting, overlap-aware broadness diagnostics, flat-baseline-consistent proposal detection, more visible Fourier status/results, and component ownership exports. Peaks may overlap, but a single component spanning more than two resolved maxima is inadmissible.

## Install

```bash
python -m pip install -e ".[gui]"
```

## Development

SIFTER supports Python 3.11 through 3.13. Install the development environment with:

```bash
python -m pip install -e ".[dev,gui]"
```

The v0.3 implementation extends the reviewed v0.2 plan in [docs/superpowers/plans/2026-09-04-sifter-v0.2.md](docs/superpowers/plans/2026-09-04-sifter-v0.2.md).

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
        fourier=True,
        random_seed=42,
        measurement_context=MeasurementContext(temperature=300.0, temperature_unit="K"),
    ),
)

print(result.best_model.peaks)
print(result.candidates[0].delta_bic)
result.to_dataframe().to_csv("sifter.fit.csv", index=False)
```

Warnings use stable codes and plain-language messages. Treat poor-resolution, broad-component, near-bound, correlation, truncation, and unavailable-uncertainty warnings as limits on interpretation, not optimizer noise to hide.

Read the [scientific method](docs/scientific-method.md), [result schema](docs/result-schema.md), [privacy policy](docs/privacy.md), and [benchmark guide](benchmarks/README.md).

## Privacy

Real experimental spectra must never be committed. Keep them in the ignored `data/` or `private/` directories. Public examples, tests, and benchmarks will use deterministic synthetic spectra only.
