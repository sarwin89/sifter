# SIFTER

**Spectral Inference using Fourier Transforms for Energy Resolution**

SIFTER is a local-first scientific Python project for reproducible decomposition of one-dimensional spectra. Version 0.1 will fit Gaussian, Lorentzian, and Voigt peak models in the original data domain while using Fourier-domain information only as an auxiliary diagnostic and initializer.

The project is currently in its architecture phase. The approved v0.1 design is documented in [docs/superpowers/specs/2026-09-03-sifter-v0.1-design.md](docs/superpowers/specs/2026-09-03-sifter-v0.1-design.md).

## Development

SIFTER supports Python 3.11 through 3.13. Install the development environment with:

```bash
python -m pip install -e ".[dev,gui]"
```

The implementation follows the reviewed tasks in [docs/superpowers/plans/2026-09-03-sifter-v0.1.md](docs/superpowers/plans/2026-09-03-sifter-v0.1.md).

## Privacy

Real experimental spectra must never be committed. Keep them in the ignored `data/` or `private/` directories. Public examples, tests, and benchmarks will use deterministic synthetic spectra only.
