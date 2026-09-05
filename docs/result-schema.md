# Result schema

`FitResult.schema_version` is `sifter.fit_result.v2` for SIFTER 0.2. Explicit legacy
`sifter.fit_result.v1` serialization remains compatibility-tested and omits v2-only
top-level measurement and reference blocks.

The result records complete analysis settings and seed, privacy-safe input metadata, optional measurement context, optional fit-reference provenance, input axes, the recommended model, all candidate score rows, optional Fourier diagnostics, residual diagnostics, parameter uncertainty, structured warnings, observation count, and SIFTER version.

The recommended model contains its family, peak count, baseline order, canonical parameter names and bounds, center-sorted peak parameters, fitted/baseline/component/residual arrays, RSS, RMSE, AICc, BIC, parameter count, observation count, and reduced chi-squared when defined.

Warnings contain stable `code`, `severity`, `message`, and `context` fields. Consumers should branch on `code`, not message text. Unavailable values are JSON `null`. If a non-finite internal value must be omitted, serialization adds `NONFINITE_VALUE_OMITTED`; JSON never emits `NaN` or `Infinity`.

Candidate rows can have status `valid`, `failed`, or `inadmissible`. By default, a candidate whose component spans multiple resolved maxima is marked `inadmissible` with `COMPONENT_SPANS_MULTIPLE_MAXIMA` and is excluded from ordinary BIC ranking. The advanced broad-band override records `BROAD_MULTIMAX_COMPONENT_ALLOWED` when such a candidate remains rankable.

Measurement context stores temperature in kelvin, laser power in watts, and optional scalar conditions. It is provenance only: adding context does not change single-spectrum fitted parameters. A `FitReference` records trusted previous peak starts used as additional candidates. Detection, staged search, global scoring, and BIC ranking still run independently, so a wrong reference cannot force the selected model.

`to_dataframe()` returns one flat row per fitted peak. `to_json()` is deterministic, sorted, standards-compliant JSON. `plot()` returns named Plotly figures for the fit, residuals, and Fourier diagnostics when requested.
