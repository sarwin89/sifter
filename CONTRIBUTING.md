# Contributing to SIFTER

Use Python 3.11–3.13 and install `.[dev,gui]`. Work in a focused development branch and keep commits reviewable.

All behavioral changes follow red–green testing: add a failing contract, run it to confirm the intended failure, implement the smallest coherent change, then run Ruff, mypy, and the complete test suite. Scientific changes need analytic or repeated-seed quantitative evidence, not screenshots or a single favorable fit.

Fixtures, examples, screenshots, and benchmarks must be synthetic and deterministic. Never inspect or commit real spectra. Keep private inputs and generated outputs in the ignored roots documented in [docs/privacy.md](docs/privacy.md).

Before opening a pull request, run:

```bash
python -m ruff check .
python -m mypy src/sifter
python -m pytest --cov=sifter --cov-report=term-missing --cov-fail-under=90
python -m build
```

Document line-shape conventions, likelihood changes, warning-code changes, and schema compatibility. Do not present delta BIC as probability or Fourier candidate spacing as confirmed peak count.
