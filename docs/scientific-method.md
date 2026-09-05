# Scientific method

SIFTER decomposes a finite, one-dimensional spectrum into a low-order polynomial baseline and one shared family of Gaussian, Lorentzian, or Voigt peaks. Every peak amplitude is integrated area. Gaussian width is standard deviation `sigma`; Lorentzian width is half-width at half-maximum `gamma`; Voigt peaks expose both.

The governing rule is: **Fourier domain informs the fit; original data determines the fit.** Fourier magnitude-envelope tendencies and candidate spacings can initialize or diagnose candidates, but they never replace the observed intensities in optimization or scoring. Ordinary FFT analysis fails closed on materially nonuniform grids unless diagnostic-only interpolation is explicitly enabled.

Candidates include simpler peak counts and polynomial baseline orders 0–2. Fast, Standard, and Thorough modes screen bounded candidate pools before refining finalists; Exhaustive mode fits every configured count directly. Windowed progressive fitting can propose starts from local cores and halos, but local scores and baselines are not exported as global evidence. Every recommended model is jointly refined and scored against the complete original spectrum with one global baseline.

BIC selects the recommendation; AICc remains visible. Delta BIC expresses relative criterion distance, not a probability. Reduced chi-squared is reported only when positive measurement standard deviations are supplied.

Fast uncertainty uses a local Jacobian covariance approximation and withholds intervals when rank or variance assumptions fail. Thorough uncertainty resamples residuals and refits the selected model. Structured warnings identify collapsed areas, near-bound parameters, extreme sensitivity correlation, truncated peaks, close centers, ambiguous selection, and unavailable uncertainty.

By default, a fitted component must not span more than one statistically resolved maximum. Persistent broad-component violations are marked inadmissible for ordinary ranking. The explicit broad-band override is reserved for known physical bands and records that choice in the result.

Temperature, laser power, generic conditions, and trusted references are provenance or initialization aids only. They do not apply hidden heating, normalization, linewidth physics, or forced peak identity. Related-spectrum plots summarize independently fitted results by conservative center/FWHM matching and report ambiguity instead of fitting a global condition-dependent physical model.

Version 0.2 is validated on deterministic, identifiable synthetic cases. It makes no claim of super-resolution, technique-specific physics, causal peak assignment, or reliable recovery when the data do not identify a unique decomposition.
