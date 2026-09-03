# Privacy and local operation

SIFTER is local-first. Spectrum content never leaves the machine: the package has no telemetry, cloud backend, account system, or network upload path. The Streamlit server binds to `localhost`.

Keep experimental inputs under ignored `data/` or `private/` directories. Keep generated analyses under ignored `results/`, `outputs/`, `reports/`, or `benchmark_results/`. Files matching `*.experimental.csv`, `*.fit.json`, `*.fit.csv`, and `*.fit.png` are ignored by design.

Public tests, examples, screenshots, and benchmarks must use deterministic synthetic spectra. Serialized result metadata strips directory components from path/file fields and retains only a basename. Do not place credentials, sample identifiers, full paths, or copied data rows in metadata, logs, issues, or commits.

Exports are created only after an explicit API call or download action. Review files before sharing them: numerical arrays are intentionally included in the versioned JSON result for reproducibility.
