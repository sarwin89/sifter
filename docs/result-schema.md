# Result schema

`FitResult.schema_version` is `sifter.fit_result.v2` for SIFTER 0.4. Explicit legacy
`sifter.fit_result.v1` serialization remains compatibility-tested and omits v2-only
top-level measurement and reference blocks.

The result records complete analysis settings and seed, manual peak-center hints, privacy-safe input metadata, optional measurement context, optional fit-reference provenance, input axes, the recommended model, retained top candidate models for visualization, all candidate score rows, optional Fourier diagnostics, residual diagnostics, parameter uncertainty, structured warnings, observation count, and SIFTER version.

Each retained model contains its family, peak count, baseline order, canonical parameter names and bounds, center-sorted peak parameters, fitted/baseline/component/residual arrays, RSS, RMSE, AICc, BIC, parameter count, observation count, reduced chi-squared when defined, and component diagnostics.

Warnings contain stable `code`, `severity`, `message`, and `context` fields. Consumers should branch on `code`, not message text. Unavailable values are JSON `null`. If a non-finite internal value must be omitted, serialization adds `NONFINITE_VALUE_OMITTED`; JSON never emits `NaN` or `Infinity`.

Candidate rows can have status `valid`, `failed`, or `inadmissible`. Component diagnostics record each component's effective support, resolved maxima inside that support, and admissibility state. A component spanning two resolved maxima remains rankable with `COMPONENT_SPANS_TWO_MAXIMA`; a component spanning more than two resolved maxima is marked `inadmissible` with `COMPONENT_SPANS_MORE_THAN_TWO_MAXIMA` and is excluded from BIC ranking.

Measurement context stores temperature in kelvin, laser power in watts, and optional scalar conditions. It is provenance only: adding context does not change single-spectrum fitted parameters. A `FitReference` records trusted previous peak starts used as additional candidates. Detection, staged search, global scoring, and BIC ranking still run independently, so a wrong reference cannot force the selected model.

`to_peak_table()` returns the user-facing peak measurement table: peak number, location, height, FWHM width, area, fit note, maxima spanned, local area error, and height error. `to_dataframe()` remains the technical flat parameter table. `to_json()` is deterministic, sorted, standards-compliant JSON. `plot()` returns named Plotly figures for the fit, residuals, and Fourier diagnostics when requested, with optional display downsampling and component-trace control.
