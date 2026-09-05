"""SIFTER's local-only Streamlit adapter."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from sifter import (
    AnalysisError,
    AutofitConfig,
    FitResult,
    MeasurementContext,
    ProgressEvent,
    SpectrumPreview,
    autofit,
    preview_spectrum,
    summarize_related_spectra,
)
from sifter.io import DelimiterOption, HeaderMode, load_spectrum, preview_table
from sifter.plotting import render_fit_png

SHAPE_LABELS = {
    "Gaussian": "gaussian",
    "Lorentzian": "lorentzian",
    "Voigt": "voigt",
}
DELIMITER_LABELS: dict[str, DelimiterOption] = {
    "Auto": "auto",
    "Tab": "\t",
    "Comma": ",",
    "Semicolon": ";",
    "Whitespace": "whitespace",
}
HEADER_LABELS: dict[str, HeaderMode] = {
    "Auto": "auto",
    "Present": "present",
    "Absent": "absent",
}
SEARCH_MODE_LABELS = {
    "Fast": "fast",
    "Standard": "standard",
    "Thorough": "thorough",
    "Exhaustive": "exhaustive",
}
PHASE_LABELS = {
    "preprocessing": "Preparing spectrum",
    "screening": "Screening candidate models",
    "expansion": "Expanding the peak-count search",
    "refinement": "Refining finalists on the full spectrum",
    "final_fitting": "Fitting candidate models",
    "uncertainty": "Estimating uncertainty",
    "completion": "Analysis complete",
}


def main() -> None:
    st.set_page_config(page_title="SIFTER", page_icon="◌", layout="wide")
    st.session_state.setdefault("fit_history", [])
    _styles()
    st.markdown('<p class="eyebrow">LOCAL SPECTRAL INFERENCE</p>', unsafe_allow_html=True)
    st.title("SIFTER")
    st.markdown(
        '<p class="lede">Decompose one-dimensional spectra with conservative model '
        "comparison and Fourier-assisted diagnostics.</p>",
        unsafe_allow_html=True,
    )
    st.info("Your spectrum stays on this machine. SIFTER sends no data or telemetry.")

    st.markdown("### 01 · Load spectrum")
    uploaded = st.file_uploader(
        "Choose a CSV, TXT, or DAT file",
        type=["csv", "txt", "dat"],
        help="Comma, tab, semicolon, and whitespace separators are detected locally.",
    )
    if uploaded is None:
        st.caption("Start with a table containing one coordinate column and one intensity column.")
        return

    with st.expander("Import options", expanded=False):
        import_columns = st.columns(3)
        with import_columns[0]:
            delimiter_label = st.selectbox(
                "Delimiter",
                tuple(DELIMITER_LABELS),
                key="input_delimiter",
                help="Use an explicit delimiter if automatic detection reports malformed rows.",
            )
        with import_columns[1]:
            header_label = st.selectbox(
                "Header",
                tuple(HEADER_LABELS),
                key="input_header",
            )
        with import_columns[2]:
            skip_rows = int(
                st.number_input(
                    "Leading rows to skip",
                    min_value=0,
                    value=0,
                    step=1,
                    key="input_skip_rows",
                    help="Skip instrument titles or metadata lines before the table header.",
                )
            )
    delimiter = DELIMITER_LABELS[delimiter_label]
    header = HEADER_LABELS[header_label]

    try:
        preview = preview_table(
            uploaded,
            delimiter=delimiter,
            header=header,
            skip_rows=skip_rows,
        )
    except (TypeError, ValueError) as error:
        st.error(f"SIFTER could not preview this table. {error}")
        return
    for warning in preview.warnings:
        st.warning(f"{warning}: Confirm the inferred columns before analysis.")
    delimiter_name = "whitespace" if preview.delimiter == "whitespace" else repr(preview.delimiter)
    st.caption(
        f"Detected {len(preview.columns)} columns · delimiter {delimiter_name} · "
        f"header {'present' if preview.has_header else 'absent'}"
    )
    st.dataframe(
        pd.DataFrame(preview.rows[:8], columns=preview.columns),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### 02 · Map columns")
    mapping_columns = st.columns(3)
    with mapping_columns[0]:
        x_column = st.selectbox("Coordinate", preview.columns, key="x_column")
    with mapping_columns[1]:
        intensity_default = min(1, len(preview.columns) - 1)
        intensity_column = st.selectbox(
            "Intensity",
            preview.columns,
            index=intensity_default,
            key="intensity_column",
        )
    with mapping_columns[2]:
        sigma_choice = st.selectbox(
            "Standard deviation (optional)",
            ("None",) + preview.columns,
            key="sigma_column",
        )

    st.markdown("### 03 · Configure analysis")
    with st.form("analysis_settings"):
        controls = st.columns((1, 1, 2, 1))
        with controls[0]:
            max_peaks = int(
                st.number_input(
                    "Maximum peaks",
                    min_value=1,
                    max_value=10,
                    value=10,
                    step=1,
                    key="max_peaks",
                )
            )
        with controls[1]:
            search_mode_label = st.selectbox(
                "Search mode",
                tuple(SEARCH_MODE_LABELS),
                index=1,
                key="search_mode",
                help="Standard screens a detector-centered search; Exhaustive fits every count.",
            )
        with controls[2]:
            selected_shapes = st.multiselect(
                "Peak shapes",
                tuple(SHAPE_LABELS),
                default=tuple(SHAPE_LABELS),
                key="shapes",
            )
        with controls[3]:
            fourier_enabled = st.checkbox(
                "Fourier assistance",
                value=True,
                key="fourier_enabled",
            )
        selected_baselines = [0, 1, 2]
        allow_fft_interpolation = False
        uncertainty_mode = "covariance"
        bootstrap_samples = 250
        random_seed = 42
        workers = 1
        allow_broad_multimax_component = False
        temperature_enabled = False
        temperature_value = 300.0
        temperature_unit = "K"
        laser_enabled = False
        laser_power = 1.0
        laser_power_unit = "mW"
        condition_name = ""
        condition_value = ""
        with st.expander("Advanced settings", expanded=False):
            selected_baselines = st.multiselect(
                "Baseline polynomial orders",
                [0, 1, 2],
                default=[0, 1, 2],
                key="baselines",
            )
            allow_fft_interpolation = st.checkbox(
                "Allow diagnostic-only interpolation for nonuniform grids",
                value=False,
                key="allow_fft_interpolation",
            )
            uncertainty_label = st.selectbox(
                "Uncertainty method",
                ("Covariance (fast)", "Bootstrap (thorough)"),
                key="uncertainty_mode",
            )
            uncertainty_mode = (
                "bootstrap" if uncertainty_label.startswith("Bootstrap") else "covariance"
            )
            bootstrap_samples = int(st.selectbox("Bootstrap fits", (100, 250, 1000), index=1))
            random_seed = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    value=42,
                    step=1,
                    key="random_seed",
                )
            )
            workers = int(
                st.number_input(
                    "Process workers",
                    min_value=1,
                    max_value=max(1, min(8, os.cpu_count() or 1)),
                    value=1,
                    step=1,
                    key="workers",
                    help="Parallelizes independent candidate models with deterministic seeds.",
                )
            )
            allow_broad_multimax_component = st.checkbox(
                "Allow known broad band across resolved maxima",
                value=False,
                key="allow_broad_multimax_component",
                help=(
                    "Records an explicit override and allows otherwise "
                    "structural-violation candidates."
                ),
            )
            st.markdown("Measurement context")
            context_columns = st.columns(2)
            with context_columns[0]:
                temperature_enabled = st.checkbox(
                    "Record temperature",
                    value=False,
                    key="record_temperature",
                )
                temperature_value = float(
                    st.number_input(
                        "Temperature",
                        value=300.0,
                        step=1.0,
                        key="temperature_value",
                    )
                )
                temperature_unit = st.selectbox(
                    "Temperature unit",
                    ("K", "C"),
                    key="temperature_unit",
                )
            with context_columns[1]:
                laser_enabled = st.checkbox(
                    "Record laser power",
                    value=False,
                    key="record_laser_power",
                )
                laser_power = float(
                    st.number_input(
                        "Laser power",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        key="laser_power_value",
                    )
                )
                laser_power_unit = st.selectbox(
                    "Laser power unit",
                    ("nW", "uW", "microW", "mW", "W"),
                    index=3,
                    key="laser_power_unit",
                )
            generic_columns = st.columns(2)
            with generic_columns[0]:
                condition_name = st.text_input(
                    "Condition name",
                    key="condition_name",
                    help="Optional label such as sample, polarization, or acquisition series.",
                )
            with generic_columns[1]:
                condition_value = st.text_input(
                    "Condition value",
                    key="condition_value",
                )
        estimated_candidates = max_peaks * len(selected_shapes) * len(selected_baselines)
        st.caption(f"Search ceiling: {estimated_candidates} candidate fits · seed {random_seed}")
        if uncertainty_mode == "bootstrap":
            st.warning(
                f"Thorough uncertainty adds {bootstrap_samples} refits after model selection."
            )
        submitted = st.form_submit_button(
            "Analyze spectrum",
            type="primary",
            key="analyze",
            disabled=not selected_shapes or not selected_baselines,
        )

    try:
        spectrum = load_spectrum(
            uploaded,
            x_column=x_column,
            intensity_column=intensity_column,
            sigma_column=None if sigma_choice == "None" else sigma_choice,
            delimiter=delimiter,
            header=header,
            skip_rows=skip_rows,
        )
        spectrum_preview = preview_spectrum(
            spectrum,
            config=AutofitConfig(
                max_peaks=max_peaks,
                fourier=fourier_enabled,
                interpolate_nonuniform_fft=allow_fft_interpolation,
            ),
        )
    except (TypeError, ValueError) as error:
        st.error(f"SIFTER could not prepare this spectrum. {error}")
        return
    _render_preview(spectrum_preview)

    if submitted:
        progress_bar = st.progress(0, text="Preparing analysis…")
        fit_status = st.status("Preparing analysis…", expanded=False)

        def report_progress(event: ProgressEvent) -> None:
            label = _progress_label(event)
            progress_bar.progress(_overall_progress(event), text=label)
            fit_status.update(
                label=label,
                state="complete" if event.phase == "completion" else "running",
            )

        try:
            config = AutofitConfig(
                max_peaks=max_peaks,
                search_mode=SEARCH_MODE_LABELS[search_mode_label],
                shapes=tuple(SHAPE_LABELS[label] for label in selected_shapes),
                baseline_orders=tuple(selected_baselines),
                fourier=fourier_enabled,
                interpolate_nonuniform_fft=allow_fft_interpolation,
                uncertainty=uncertainty_mode,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
                workers=workers,
                allow_broad_multimax_component=allow_broad_multimax_component,
                measurement_context=_measurement_context(
                    temperature_enabled=temperature_enabled,
                    temperature_value=temperature_value,
                    temperature_unit=temperature_unit,
                    laser_enabled=laser_enabled,
                    laser_power=laser_power,
                    laser_power_unit=laser_power_unit,
                    condition_name=condition_name,
                    condition_value=condition_value,
                ),
            )
            result = autofit(
                spectrum,
                config=config,
                progress=report_progress,
            )
            st.session_state["fit_result"] = result
            history = st.session_state.setdefault("fit_history", [])
            history.append(result)
            st.session_state["fit_history"] = history[-12:]
        except (AnalysisError, TypeError, ValueError) as error:
            st.session_state.pop("fit_result", None)
            fit_status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"SIFTER could not complete the analysis. {_analysis_error_message(error)}")

    result = st.session_state.get("fit_result")
    if isinstance(result, FitResult):
        _render_result(result)
    history = tuple(
        item for item in st.session_state.get("fit_history", ()) if isinstance(item, FitResult)
    )
    if len(history) >= 2:
        _render_related(history)


def _render_result(result: FitResult) -> None:
    model = result.best_model
    st.markdown("### 05 · Inspect and export")
    st.subheader(
        f"Recommended model · {model.peak_count} {model.shape.title()} "
        f"{'peak' if model.peak_count == 1 else 'peaks'}"
    )
    metrics = st.columns(4)
    metrics[0].metric("BIC", f"{model.bic:.2f}")
    metrics[1].metric("AICc", f"{model.aicc:.2f}")
    metrics[2].metric("RMSE", f"{model.rmse:.4g}")
    metrics[3].metric("Seed", str(result.settings.random_seed))
    method = "Covariance" if result.uncertainty.method == "covariance" else "Bootstrap"
    st.caption(f"{method} uncertainty · deterministic seed {result.settings.random_seed}")
    for warning in result.warnings:
        st.warning(f"{warning.code}: {warning.message}")

    for name, figure in result.plot().items():
        if name == "fourier" and result.fourier is not None and result.fourier.frequency.size == 0:
            continue
        st.plotly_chart(figure, width="stretch", key=f"result_{name}")

    _render_result_fourier(result)

    st.subheader("Candidate comparison")
    rows = [
        {
            "Shape": score.shape.title(),
            "Peaks": score.peak_count,
            "Baseline": score.baseline_order,
            "Status": score.status,
            "BIC": score.bic,
            "ΔBIC": score.delta_bic,
            "AICc": score.aicc,
            "Failure": score.failure_code,
        }
        for score in result.candidates
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download result JSON",
        result.to_json(),
        file_name="sifter.fit.json",
        mime="application/json",
    )
    downloads[1].download_button(
        "Download peak table CSV",
        result.to_dataframe().to_csv(index=False),
        file_name="sifter.fit.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "Download fit PNG",
        render_fit_png(result),
        file_name="sifter.fit.png",
        mime="image/png",
    )


def _render_result_fourier(result: FitResult) -> None:
    st.subheader("Fourier diagnostics")
    if result.fourier is None:
        st.caption("Fourier diagnostics were disabled for this analysis.")
        return

    fourier = result.fourier
    if fourier.frequency.size == 0:
        st.warning(
            "Fourier diagnostics unavailable: "
            f"{fourier.warning_code or 'INSUFFICIENT_FOURIER_RANGE'}"
        )
        st.caption(
            "For nonuniform coordinate grids, enable diagnostic-only interpolation "
            "in Advanced settings if an FFT preview is acceptable. Fits and model "
            "scores still use the original observations."
        )
        return

    status = "available" if fourier.applicable else "limited"
    interpolation = "interpolated grid" if fourier.interpolated else "native uniform grid"
    st.caption(
        f"Fourier diagnostics {status} · {fourier.window.title()} window · {interpolation}"
    )
    if fourier.warning_code is not None:
        st.warning(f"{fourier.warning_code}: Fourier diagnostics are reported with caution.")

    if fourier.candidate_spacings:
        st.table(
            {
                "Candidate spacings": ", ".join(
                    f"{spacing:.4g}" for spacing in fourier.candidate_spacings
                )
            },
            border="horizontal",
            width="content",
        )

    if fourier.envelope_fits:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Family": fit.family.title(),
                        "BIC": fit.bic,
                        "RSS": fit.rss,
                        "Frequency min": fit.frequency_min,
                        "Frequency max": fit.frequency_max,
                    }
                    for fit in fourier.envelope_fits
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_related(history: tuple[FitResult, ...]) -> None:
    st.markdown("### 06 · Related spectra")
    choices = _related_condition_choices(history)
    if not choices:
        st.caption("Record temperature, laser power, or a generic condition to compare spectra.")
        return
    condition = st.selectbox(
        "Condition axis",
        choices,
        key="related_condition_axis",
    )
    table = summarize_related_spectra(history, condition=condition)
    st.dataframe(table, width="stretch", hide_index=True)
    if table.empty or table["condition_value"].isna().all():
        st.caption("No numeric condition values are available for plotting.")
        return
    numeric = table.copy()
    numeric["condition_value"] = pd.to_numeric(numeric["condition_value"], errors="coerce")
    numeric = numeric.dropna(subset=["condition_value"])
    if numeric.empty:
        st.caption("The selected condition is nonnumeric, so only the table is shown.")
        return
    for value in ("center", "fwhm", "area"):
        st.line_chart(
            numeric,
            x="condition_value",
            y=value,
            color="track_id",
            x_label=condition.replace("_", " ").title(),
            y_label=value.upper() if value == "fwhm" else value.title(),
        )


def _related_condition_choices(history: tuple[FitResult, ...]) -> tuple[str, ...]:
    choices: list[str] = []
    for result in history:
        context = result.measurement_context
        if context is None:
            continue
        if context.temperature is not None and "temperature" not in choices:
            choices.append("temperature")
        if context.laser_power is not None and "laser_power" not in choices:
            choices.append("laser_power")
        if context.conditions is not None:
            for key in context.conditions:
                if key not in choices:
                    choices.append(key)
    return tuple(choices)


def _measurement_context(
    *,
    temperature_enabled: bool,
    temperature_value: float,
    temperature_unit: str,
    laser_enabled: bool,
    laser_power: float,
    laser_power_unit: str,
    condition_name: str,
    condition_value: str,
) -> MeasurementContext | None:
    conditions = {}
    clean_name = condition_name.strip()
    if clean_name:
        conditions[clean_name] = condition_value.strip()
    if not temperature_enabled and not laser_enabled and not conditions:
        return None
    return MeasurementContext(
        temperature=temperature_value if temperature_enabled else None,
        temperature_unit=temperature_unit if temperature_enabled else None,
        laser_power=laser_power if laser_enabled else None,
        laser_power_unit=laser_power_unit if laser_enabled else None,
        conditions=conditions,
    )


def _render_preview(preview: SpectrumPreview) -> None:
    st.markdown("### 04 · Pre-fit preview")
    grid_label = "Uniform FFT grid" if preview.grid_is_uniform else "Nonuniform FFT grid"
    interpolation_label = (
        "diagnostic interpolation enabled"
        if preview.fourier_interpolated
        else "no diagnostic interpolation"
    )
    st.caption(
        f"{grid_label} · median step {preview.grid_median_step:.4g} "
        f"{preview.x_unit or preview.x_name} · {interpolation_label}"
    )
    real_space = pd.DataFrame(
        {
            preview.x_name: preview.x,
            "Raw intensity": preview.intensity,
            "Global baseline": preview.baseline,
            "Baseline-adjusted": preview.adjusted,
        }
    )
    st.line_chart(
        real_space,
        x=preview.x_name,
        y=["Raw intensity", "Global baseline", "Baseline-adjusted"],
        x_label=(
            preview.x_name
            if preview.x_unit is None
            else f"{preview.x_name} ({preview.x_unit})"
        ),
        y_label=preview.intensity_name,
    )
    if preview.provisional_centers:
        proposals = pd.DataFrame(
            {
                "Center": preview.provisional_centers,
                "Width estimate": preview.provisional_widths,
                "Prominence": preview.provisional_prominences,
            }
        )
        st.dataframe(proposals, width="stretch", hide_index=True)
    else:
        st.caption("No provisional centers passed the conservative detector thresholds.")

    if not preview.fourier_enabled:
        st.caption("Fourier diagnostics are disabled for this analysis.")
        return
    if preview.frequency.size == 0:
        st.warning(
            f"Fourier preview unavailable: {preview.fourier_warning_code or 'insufficient data'}"
        )
        return

    fourier_columns = st.columns(2)
    magnitude = pd.DataFrame(
        {"Frequency": preview.frequency, "FFT magnitude": preview.magnitude}
    )
    with fourier_columns[0]:
        st.line_chart(
            magnitude,
            x="Frequency",
            y="FFT magnitude",
            x_label=f"Frequency ({preview.frequency_unit})",
            y_label="Magnitude",
        )
    log_data: dict[str, object] = {
        "Frequency": preview.frequency,
        "Log magnitude": preview.log_magnitude,
    }
    for envelope in preview.envelope_fits:
        frequency = preview.frequency
        if envelope.family == "gaussian":
            tendency = envelope.intercept - envelope.decay_coefficients[0] * frequency**2
        elif envelope.family == "lorentzian":
            tendency = envelope.intercept - envelope.decay_coefficients[0] * frequency
        else:
            tendency = (
                envelope.intercept
                - envelope.decay_coefficients[0] * frequency
                - envelope.decay_coefficients[1] * frequency**2
            )
        log_data[f"{envelope.family.title()} tendency"] = tendency
    with fourier_columns[1]:
        st.line_chart(
            pd.DataFrame(log_data),
            x="Frequency",
            y=[column for column in log_data if column != "Frequency"],
            x_label=f"Frequency ({preview.frequency_unit})",
            y_label="Log magnitude",
        )
    spacings = ", ".join(f"{value:.4g}" for value in preview.candidate_spacings)
    st.caption(
        f"{preview.fourier_window.title() if preview.fourier_window else 'No'} window · "
        f"candidate spacings: {spacings or 'none'} {preview.x_unit or preview.x_name}"
    )


def _overall_progress(event: ProgressEvent) -> int:
    ranges = {
        "preprocessing": (0, 10),
        "screening": (10, 45),
        "expansion": (45, 55),
        "refinement": (55, 80),
        "final_fitting": (10, 80),
        "uncertainty": (80, 98),
        "completion": (100, 100),
    }
    start, end = ranges[event.phase]
    fraction = 1.0 if event.total == 0 else event.completed / event.total
    return round(start + (end - start) * fraction)


def _progress_label(event: ProgressEvent) -> str:
    label = PHASE_LABELS[event.phase]
    if event.total > 0:
        label = f"{label} · {event.completed}/{event.total}"
    if event.message:
        label = f"{label} · {event.message}"
    return label


def _analysis_error_message(error: Exception) -> str:
    if isinstance(error, AnalysisError) and error.code == "NO_RANKABLE_CANDIDATE":
        rejection_counts = (
            pd.Series(
                [
                    score.failure_code or score.status
                    for score in error.candidate_scores
                ]
            )
            .value_counts()
            .to_dict()
        )
        common = ", ".join(
            f"{code}: {count}" for code, count in rejection_counts.items()
        )
        advice = (
            "No candidate remained rankable after validation. "
            "If this is a known broad physical band, enable the advanced broad-band "
            "override; otherwise increase the peak limit or narrow the fitted range."
        )
        return f"{advice} Rejections: {common or 'none'}."
    return str(error)


def _styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: oklch(96% 0.014 85);
          --ink: oklch(25% 0.026 165);
          --mineral: oklch(45% 0.09 165);
          --line: oklch(81% 0.026 120);
        }
        .stApp {
          background: #f5f2e9;
          background: var(--paper);
          color: #17231f;
          color: var(--ink);
          font-family: Aptos, "Trebuchet MS", sans-serif;
        }
        [data-testid="stMainBlockContainer"] { max-width: 86rem; padding-top: 3rem; }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: var(--ink); }
        h1 { font-size: 4rem; letter-spacing: -0.055em; margin: 0; }
        .eyebrow { color: var(--mineral); font-size: .78rem; font-weight: 700;
                   letter-spacing: .18em; margin-bottom: .2rem; }
        .lede { font-family: Georgia, "Times New Roman", serif; font-size: 1.3rem;
                line-height: 1.5; max-width: 58ch; margin: 0 0 2rem; color: var(--ink); }
        [data-testid="stMetricValue"], [data-testid="stDataFrame"] {
          font-variant-numeric: tabular-nums;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] p,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
          color: #29483d !important;
        }
        div[data-baseweb="select"] > div,
        [data-testid="stNumberInputContainer"] {
          background: #fbfaf5 !important;
          border-color: #9aaa9f !important;
        }
        div[data-baseweb="select"] *,
        [data-testid="stNumberInputContainer"] input {
          color: #17231f !important;
        }
        span[data-baseweb="tag"] { background: #2f705d !important; }
        span[data-baseweb="tag"] * { color: #f8f5ec !important; }
        [data-testid="stCheckbox"] [data-checked="true"] {
          background-color: #176b55 !important;
          border-color: #176b55 !important;
        }
        div[data-testid="stFileUploaderDropzone"] { border-color: var(--line); }
        .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
          background: #176b55 !important;
          border-color: #176b55 !important;
          min-height: 2.75rem;
          border-radius: .2rem;
        }
        button[kind="primary"] p { color: #f8f5ec !important; }
        @media (max-width: 700px) {
          h1 { font-size: 3rem; }
          [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
