# SIFTER benchmarks

These public benchmarks use deterministic synthetic spectra with known ground truth. Functions return pandas tables and do not write files unless an explicit `output_path` is supplied. Generated results belong in the ignored `benchmark_results/` directory.

Quick CI checks use two seeds, one or two well-separated Gaussian peaks, and a single Gaussian width case. They verify the pipeline and catch regressions; they are not performance or super-resolution claims.

A thorough study should sweep the full resolution grid across separation/FWHM, SNR, amplitude ratio, Voigt fraction, sampling density, baseline slope, and repeated seeds. Report aggregate recovery rates and error distributions, not a favorable single spectrum.

Candidate-level runtime profiling includes fixed, well-resolved three- and ten-peak
Gaussian cases. Timings are descriptive and are never used as wall-clock CI gates:

```python
from benchmarks.benchmark_candidate_runtime import benchmark_candidate_runtime

timings = benchmark_candidate_runtime(repeats=3)
timings.to_csv("benchmark_results/candidate-runtime.csv", index=False)
```

```python
from benchmarks.generate_resolution_grid import generate_resolution_grid

design = generate_resolution_grid(seeds=tuple(range(20)))
design.to_csv("benchmark_results/resolution-design.csv", index=False)
```
