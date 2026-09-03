# SIFTER

**Spectral Inference using Fourier Transforms for Energy Resolution**

SIFTER is a local-first scientific Python project for reproducible decomposition of one-dimensional spectra. Version 0.1 will fit Gaussian, Lorentzian, and Voigt peak models in the original data domain while using Fourier-domain information only as an auxiliary diagnostic and initializer.

Fourier evidence initializes and diagnoses fits; only the original observations determine parameters and model scores. SIFTER reports uncertainty and identifiability limitations rather than claiming resolution the data cannot support.

## Install

```bash
python -m pip install -e ".[gui]"
```

## Development

SIFTER supports Python 3.11 through 3.13. Install the development environment with:

```bash
python -m pip install -e ".[dev,gui]"
```

The implementation follows the reviewed tasks in [docs/superpowers/plans/2026-09-03-sifter-v0.1.md](docs/superpowers/plans/2026-09-03-sifter-v0.1.md).

## Local GUI

Install the GUI dependencies and launch the local application:

```bash
python -m pip install -e ".[gui]"
sifter-gui
```

The browser interface runs on `localhost`; SIFTER does not upload spectra or send telemetry.

## Python API

```python
from sifter import Spectrum, autofit

spectrum = Spectrum(x, intensity, x_name="Raman shift", x_unit="cm⁻¹")
result = autofit(
    spectrum,
    max_peaks=6,
    shapes=("gaussian", "lorentzian", "voigt"),
    fourier=True,
    random_seed=42,
)

print(result.best_model.peaks)
print(result.candidates[0].delta_bic)
result.to_dataframe().to_csv("sifter.fit.csv", index=False)
```

Warnings use stable codes and plain-language messages. Treat poor-resolution, near-bound, correlation, truncation, and unavailable-uncertainty warnings as limits on interpretation—not optimizer noise to hide.

Read the [scientific method](docs/scientific-method.md), [result schema](docs/result-schema.md), [privacy policy](docs/privacy.md), and [benchmark guide](benchmarks/README.md).

## Privacy

Real experimental spectra must never be committed. Keep them in the ignored `data/` or `private/` directories. Public examples, tests, and benchmarks will use deterministic synthetic spectra only.
