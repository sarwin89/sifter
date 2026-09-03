# Result schema

`FitResult.schema_version` is exactly `sifter.fit_result.v1` for SIFTER 0.1.

The result records complete analysis settings and seed, privacy-safe input metadata, input axes, the recommended model, all candidate score rows, optional Fourier diagnostics, residual diagnostics, parameter uncertainty, structured warnings, observation count, and SIFTER version.

The recommended model contains its family, peak count, baseline order, canonical parameter names and bounds, center-sorted peak parameters, fitted/baseline/component/residual arrays, RSS, RMSE, AICc, BIC, parameter count, observation count, and reduced chi-squared when defined.

Warnings contain stable `code`, `severity`, `message`, and `context` fields. Consumers should branch on `code`, not message text. Unavailable values are JSON `null`. If a non-finite internal value must be omitted, serialization adds `NONFINITE_VALUE_OMITTED`; JSON never emits `NaN` or `Infinity`.

`to_dataframe()` returns one flat row per fitted peak. `to_json()` is deterministic, sorted, standards-compliant JSON. `plot()` returns named Plotly figures for the fit, residuals, and Fourier diagnostics when requested.
