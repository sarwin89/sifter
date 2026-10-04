"""Local Streamlit workbench for SIFTER v0.4.2."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st

from sifter import FitConfig, FitResult, ModelResult, ProgressEvent, Spectrum, fit_spectrum
from sifter.config import CountMode, FitMode, PeakShape
from sifter.io import HeaderMode, load_spectrum, preview_table
from sifter.plotting import render_fit_png
from sifter.progress import ProgressPhase
from sifter.workbench import DetectedMaximum, SpectrumInspection, inspect_spectrum

DELIMITER_LABELS = {
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
SHAPE_LABELS = {"Gaussian": "gaussian", "Lorentzian": "lorentzian", "Voigt": "voigt"}
COUNT_MODE_LABELS = {"Auto": "auto", "Exact": "exact"}
FIT_MODE_LABELS = {
    "Fast": "fast",
    "Standard": "standard",
    "Thorough": "thorough",
    "Research": "research",
}
PHASE_LABELS: dict[ProgressPhase, str] = {
    "preprocessing": "Preparing spectrum",
    "screening": "Screening candidate models",
    "expansion": "Expanding candidate counts",
    "refinement": "Refining finalists",
    "final_fitting": "Fitting candidate models",
    "uncertainty": "Estimating uncertainty",
    "completion": "Completed",
}

_PEAK_PICKER = (
    st.components.v2.component(
        "sifter_peak_picker",
        html="""
<div class="picker">
  <svg class="plot" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
  <div class="hint">Click to add a peak hint. Click near an existing hint to remove it.</div>
</div>
""",
        css="""
.picker {
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius);
  background: var(--st-secondary-background-color);
  padding: 10px;
}
.plot {
  width: 100%;
  height: 280px;
  background: var(--st-background-color);
  border-radius: var(--st-base-radius);
  touch-action: manipulation;
}
.hint {
  color: var(--st-text-color);
  font: 12px var(--st-font);
  opacity: 0.75;
  margin-top: 6px;
}
""",
        js="""
export default function(component) {
  const { data, parentElement, setStateValue } = component
  const svg = parentElement.querySelector("svg.plot")
  if (!svg || !data) return
  const width = 900
  const height = 280
  const pad = 18
  const x = data.x || []
  const y = data.y || []
  const maxima = data.maxima || []
  let hints = Array.isArray(data.hints) ? [...data.hints] : []
  if (x.length < 2 || y.length < 2) return
  const xmin = Math.min(...x)
  const xmax = Math.max(...x)
  const ymin = Math.min(...y)
  const ymax = Math.max(...y)
  const xscale = value =>
    pad + (value - xmin) / Math.max(xmax - xmin, 1e-12) * (width - 2 * pad)
  const yscale = value =>
    height - pad - (value - ymin) / Math.max(ymax - ymin, 1e-12) * (height - 2 * pad)
  const points = x.map((value, index) =>
    `${xscale(value).toFixed(2)},${yscale(y[index]).toFixed(2)}`
  ).join(" ")
  const verticalLine = (value, color, width, opacity = 1) =>
    `<line x1="${xscale(value)}" x2="${xscale(value)}" y1="${pad}" ` +
    `y2="${height - pad}" stroke="${color}" stroke-width="${width}" ` +
    `opacity="${opacity}" />`
  const maximaLines = maxima.map(value =>
    verticalLine(value, "var(--st-orange-color)", "1.2", "0.65")
  ).join("")
  const hintLines = hints.map(value =>
    verticalLine(value, "var(--st-primary-color)", "2.4")
  ).join("")
  svg.innerHTML = `
    <polyline points="${points}" fill="none" stroke="var(--st-text-color)"
      stroke-width="1.4" opacity="0.82" />
    ${maximaLines}
    ${hintLines}
  `
  svg.onclick = event => {
    const rect = svg.getBoundingClientRect()
    const px = (event.clientX - rect.left) / Math.max(rect.width, 1) * width
    const value = xmin + (px - pad) / Math.max(width - 2 * pad, 1) * (xmax - xmin)
    const tolerance = (xmax - xmin) * 0.01
    const existing = hints.findIndex(item => Math.abs(item - value) <= tolerance)
    if (existing >= 0) {
      hints.splice(existing, 1)
    } else if (hints.length < data.maxHints) {
      hints.push(value)
    }
    hints = [...new Set(hints.map(item => Number(item.toFixed(8))))].sort((a, b) => a - b)
    setStateValue("hints", hints)
  }
}
""",
    )
    if hasattr(st, "components") and hasattr(st.components, "v2")
    else None
)


def main() -> None:
    st.set_page_config(page_title="SIFTER", page_icon=":material/query_stats:", layout="wide")
    st.title("SIFTER — Fast peak workbench", icon=":material/query_stats:")
    st.caption("v0.4.2 research-grade local fitting · fast preview · exact final refinement")
    st.info("Your spectrum stays on this machine. SIFTER sends no data or telemetry.")
    st.session_state.setdefault("fit_history", [])

    uploaded = st.file_uploader(
        "Load spectrum",
        type=("csv", "txt", "tsv", "dat"),
        help=(
            "Use a two-column or three-column local table. "
            "Tab, comma, semicolon, and whitespace are supported."
        ),
    )
    if uploaded is None:
        st.caption("Start by loading a table with an x-axis column and an intensity column.")
        return

    payload = uploaded.getvalue()
    import_settings = _import_settings()
    try:
        preview = preview_table(payload, **import_settings)
    except ValueError as error:
        st.error(f"SIFTER could not preview this table. {error}")
        return

    st.subheader("1 · Check imported table")
    st.dataframe(pd.DataFrame(preview.rows, columns=preview.columns).head(12), hide_index=True)
    if preview.warnings:
        st.warning(", ".join(preview.warnings))

    spectrum = _mapped_spectrum(payload, preview.columns, import_settings)
    if spectrum is None:
        return

    max_peaks = int(
        st.number_input(
            "Maximum peaks",
            min_value=1,
            max_value=10,
            value=10,
            key="max_peaks",
        )
    )
    inspection = inspect_spectrum(spectrum, max_peaks=max_peaks)
    peak_hints = _render_inspection(spectrum, inspection, max_peaks=max_peaks)
    config = _fit_controls(max_peaks=max_peaks, peak_hints=peak_hints)

    if st.button("Run fit", type="primary", icon=":material/play_arrow:", key="analyze"):
        _run_fit(spectrum, config)

    result = st.session_state.get("fit_result")
    if isinstance(result, FitResult):
        _render_result(result)


def _import_settings() -> dict[str, Any]:
    st.subheader("Import options")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        delimiter_label = st.selectbox("Delimiter", tuple(DELIMITER_LABELS), key="input_delimiter")
        header_label = st.selectbox("Header", tuple(HEADER_LABELS), key="input_header")
        skip_rows = int(
            st.number_input(
                "Skip rows",
                min_value=0,
                max_value=200,
                value=0,
                step=1,
                key="input_skip_rows",
            )
        )
    return {
        "delimiter": DELIMITER_LABELS[delimiter_label],
        "header": HEADER_LABELS[header_label],
        "skip_rows": skip_rows,
    }


def _mapped_spectrum(
    payload: bytes,
    columns: tuple[str, ...],
    import_settings: dict[str, Any],
) -> Spectrum | None:
    st.subheader("2 · Map columns")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        x_column = st.selectbox("Coordinate", columns, key="x_column")
        default_intensity = columns[1] if len(columns) > 1 else columns[0]
        intensity_column = st.selectbox(
            "Intensity",
            columns,
            index=columns.index(default_intensity),
            key="intensity_column",
        )
        sigma_options = ("None", *columns)
        sigma_choice = st.selectbox("Uncertainty", sigma_options, key="sigma_column")
    if x_column == intensity_column:
        st.error("Coordinate and intensity must be different columns.")
        return None
    try:
        return load_spectrum(
            payload,
            x_column=x_column,
            intensity_column=intensity_column,
            sigma_column=None if sigma_choice == "None" else sigma_choice,
            x_name=x_column,
            intensity_name=intensity_column,
            **import_settings,
        )
    except ValueError as error:
        st.error(f"SIFTER could not prepare this spectrum. {error}")
        return None


def _render_inspection(
    spectrum: Spectrum,
    inspection: SpectrumInspection,
    *,
    max_peaks: int,
) -> tuple[float, ...]:
    st.subheader("3 · Detected maxima and peak hints")
    metrics = st.columns(3)
    metrics[0].metric("Detected maxima", len(inspection.maxima))
    metrics[1].metric("Data points", spectrum.x.size)
    metrics[2].metric("Preview time", f"{inspection.elapsed_seconds:.2f}s")
    st.caption(
        "Click the spectrum to add peak hints. "
        "Orange lines are detected maxima; blue lines are your hints."
    )
    hints = _peak_picker(
        spectrum,
        inspection.maxima,
        current_hints=(),
        max_hints=max_peaks,
        key="peak_picker",
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        manual = st.text_input(
            "Peak hints",
            ", ".join(f"{hint:.6g}" for hint in hints),
            key="manual_peak_hints",
            help="Optional fallback: comma-separated x-axis peak centers.",
        )
        if st.button("Clear hints", icon=":material/close:", key="clear_peak_hints"):
            st.session_state.pop("peak_picker", None)
            st.rerun()
    parsed = _parse_peak_hints(manual)
    st.dataframe(pd.DataFrame(inspection.to_rows()), hide_index=True)
    return parsed[:max_peaks]


def _peak_picker(
    spectrum: Spectrum,
    maxima: tuple[DetectedMaximum, ...],
    *,
    current_hints: tuple[float, ...],
    max_hints: int,
    key: str,
) -> tuple[float, ...]:
    state = st.session_state.get(key, {})
    if isinstance(state, dict):
        current_hints = tuple(float(value) for value in state.get("hints", current_hints))
    if _PEAK_PICKER is None:
        st.warning("Interactive peak picker is unavailable in this Streamlit build.")
        return current_hints
    indices = _display_indices(spectrum.x.size, 1200)
    result = _PEAK_PICKER(
        key=key,
        data={
            "x": [float(value) for value in spectrum.x[indices]],
            "y": [float(value) for value in spectrum.intensity[indices]],
            "maxima": [float(maximum.center) for maximum in maxima],
            "hints": [float(value) for value in current_hints],
            "maxHints": max_hints,
        },
        on_hints_change=lambda: None,
    )
    hints = getattr(result, "hints", None)
    if hints is None:
        return current_hints
    return tuple(float(value) for value in hints)


def _fit_controls(*, max_peaks: int, peak_hints: tuple[float, ...]) -> FitConfig:
    st.subheader("4 · Fit intent")
    with st.form("fit_controls"):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            count_label = st.selectbox("Peak count", tuple(COUNT_MODE_LABELS), key="count_mode")
            fit_label = st.selectbox("Fit mode", tuple(FIT_MODE_LABELS), key="fit_mode")
            selected_shapes = st.multiselect(
                "Line shapes",
                tuple(SHAPE_LABELS),
                default=tuple(SHAPE_LABELS),
                key="shapes",
            )
            workers = int(
                st.number_input(
                    "Workers",
                    min_value=1,
                    max_value=8,
                    value=1,
                    key="workers",
                )
            )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            fourier = st.toggle("Fourier diagnostics", value=True, key="fourier_enabled")
            allow_fft_interpolation = st.toggle(
                "Interpolate nonuniform FFT",
                value=False,
                key="allow_fft_interpolation",
            )
            allow_broad = st.toggle(
                "Allow shoulder notes",
                value=False,
                key="allow_broad_multimax_component",
                help="Components spanning more than two maxima remain inadmissible.",
            )
        submitted = st.form_submit_button("Apply settings", icon=":material/tune:")
    del submitted
    if not selected_shapes:
        selected_shapes = ["Gaussian"]
    count_mode = cast(CountMode, COUNT_MODE_LABELS[count_label])
    fit_mode = cast(FitMode, FIT_MODE_LABELS[fit_label])
    shapes = tuple(cast(PeakShape, SHAPE_LABELS[label]) for label in selected_shapes)
    if COUNT_MODE_LABELS[count_label] == "exact" and len(peak_hints) > max_peaks:
        st.error("Exact peak count must be at least the number of peak hints.")
    return FitConfig(
        max_peaks=max_peaks,
        count_mode=count_mode,
        fit_mode=fit_mode,
        shapes=shapes,
        peak_hints=peak_hints,
        fourier=fourier,
        interpolate_nonuniform_fft=allow_fft_interpolation,
        workers=workers,
        allow_broad_multimax_component=allow_broad,
    )


def _run_fit(spectrum: Spectrum, config: FitConfig) -> None:
    progress_bar = st.progress(0, text="Preparing analysis…")
    status = st.status("Fitting spectrum…", expanded=True)

    def report(event: ProgressEvent) -> None:
        progress_bar.progress(_overall_progress(event), text=_progress_label(event))
        status.write(_progress_label(event))

    try:
        with status:
            result = fit_spectrum(spectrum, config=config, progress=report)
    except Exception as error:  # noqa: BLE001 - user-facing app boundary
        st.session_state.pop("fit_result", None)
        status.update(label="Fit failed", state="error", expanded=True)
        st.error(f"SIFTER could not complete the analysis. {error}")
        return
    st.session_state["fit_result"] = result
    history = st.session_state.setdefault("fit_history", [])
    history.append(result)
    st.session_state["fit_history"] = history[-12:]
    progress_bar.progress(100, text="Completed")
    status.update(label="Fit complete", state="complete", expanded=False)


def _render_result(result: FitResult) -> None:
    st.subheader("Recommended fit")
    selected_model = _selected_candidate_model(result)
    selected_peak_table = result.to_peak_table(model=selected_model)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Peaks", selected_model.peak_count)
    metric_columns[1].metric("Shape", selected_model.shape)
    metric_columns[2].metric(
        "Worst local error",
        _format_error(selected_peak_table["local_area_error"].max()),
    )
    metric_columns[3].metric(
        "Fit note",
        ", ".join(sorted(set(selected_peak_table["fit_note"]))),
    )

    st.subheader("Candidate explorer")
    view = st.segmented_control(
        "View",
        ("Fit", "Residuals", "Fourier", "Advanced"),
        default="Fit",
        key="result_view",
    )
    show_components = st.toggle("Show individual peak curves", value=False, key="show_components")
    if view == "Fit":
        st.line_chart(_fit_chart_frame(result, selected_model, show_components=show_components))
    elif view == "Residuals":
        st.line_chart(_residual_frame(result, selected_model))
    elif view == "Fourier":
        _render_fourier(result)
    else:
        _render_advanced(result)

    st.subheader("Peak measurements")
    st.dataframe(selected_peak_table, hide_index=True)

    st.subheader("Advanced model selection")
    st.dataframe(_candidate_frame(result), hide_index=True)

    _render_exports(result, selected_peak_table)


def _selected_candidate_model(result: FitResult) -> ModelResult:
    if not result.candidate_models:
        return result.best_model
    options = {
        f"#{index + 1}: {model.peak_count} {model.shape} peaks": model
        for index, model in enumerate(result.candidate_models)
    }
    label = st.selectbox("Candidate fit", tuple(options), key="candidate_model")
    return options[label]


def _fit_chart_frame(
    result: FitResult,
    model: ModelResult,
    *,
    show_components: bool,
) -> pd.DataFrame:
    indices = _display_indices(result.x.size, 2000)
    data: dict[str, np.ndarray] = {
        result.x_name: result.x[indices],
        "observed": result.intensity[indices],
        "fit": model.fitted[indices],
        "baseline": model.baseline[indices],
    }
    if show_components:
        for index, component in enumerate(model.components, start=1):
            data[f"peak {index}"] = component[indices] + model.baseline[indices]
    return pd.DataFrame(data).set_index(result.x_name)


def _residual_frame(result: FitResult, model: ModelResult) -> pd.DataFrame:
    indices = _display_indices(result.x.size, 2000)
    return pd.DataFrame(
        {result.x_name: result.x[indices], "residual": model.residuals[indices]}
    ).set_index(result.x_name)


def _candidate_frame(result: FitResult) -> pd.DataFrame:
    rows = []
    models = result.candidate_models or (result.best_model,)
    for index, model in enumerate(models[:10], start=1):
        peak_table = result.to_peak_table(model=model)
        notes = [str(note) for note in peak_table["fit_note"]]
        summary_note = ", ".join(sorted(set(notes))) if notes else "clear"
        rows.append(
            {
                "rank": index,
                "peak count": model.peak_count,
                "shape": model.shape,
                "worst local error": peak_table["local_area_error"].max(),
                "note count": sum(note != "clear" for note in notes),
                "summary note": summary_note,
                "BIC": model.bic,
                "AICc": model.aicc,
                "RMSE": model.rmse,
            }
        )
    return pd.DataFrame(rows)


def _render_fourier(result: FitResult) -> None:
    if result.fourier is None:
        st.caption("Fourier diagnostics were disabled for this fit.")
        return
    st.table(
        {
            "Applicable": result.fourier.applicable,
            "Interpolated": result.fourier.interpolated,
            "Warning": result.fourier.warning_code or "none",
        }
    )
    if result.fourier.frequency.size:
        indices = _display_indices(result.fourier.frequency.size, 1200)
        st.line_chart(
            pd.DataFrame(
                {
                    "frequency": result.fourier.frequency[indices],
                    "magnitude": result.fourier.magnitude[indices],
                }
            ).set_index("frequency")
        )


def _render_advanced(result: FitResult) -> None:
    st.table(
        {
            "BIC": f"{result.best_model.bic:.4g}",
            "AICc": f"{result.best_model.aicc:.4g}",
            "RMSE": f"{result.best_model.rmse:.4g}",
            "Parameters": result.best_model.parameter_count,
            "Observations": result.observation_count,
        }
    )
    for warning in result.warnings:
        st.warning(f"{warning.code}: {warning.message}")


def _render_exports(result: FitResult, peak_table: pd.DataFrame) -> None:
    st.subheader("Export")
    with st.container(horizontal=True):
        st.download_button(
            "Peak CSV",
            peak_table.to_csv(index=False).encode(),
            file_name="sifter-peaks.csv",
            mime="text/csv",
        )
        st.download_button(
            "Result JSON",
            result.to_json().encode(),
            file_name="sifter-result.json",
            mime="application/json",
        )
        st.download_button(
            "Fit PNG",
            render_fit_png(result),
            file_name="sifter-fit.png",
            mime="image/png",
        )


def _parse_peak_hints(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return ()
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            st.error(f"Peak hint {text!r} is not a number.")
            continue
        if value not in values:
            values.append(value)
    return tuple(sorted(values))


def _display_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def _overall_progress(event: ProgressEvent) -> int:
    phase_offsets = {
        "preprocessing": 5,
        "screening": 20,
        "expansion": 35,
        "refinement": 55,
        "final_fitting": 70,
        "uncertainty": 90,
        "completion": 100,
    }
    phase_width = {
        "preprocessing": 15,
        "screening": 15,
        "expansion": 20,
        "refinement": 15,
        "final_fitting": 20,
        "uncertainty": 10,
        "completion": 0,
    }
    if event.total == 0:
        return phase_offsets[event.phase]
    return min(
        100,
        phase_offsets[event.phase]
        + int(phase_width[event.phase] * event.completed / event.total),
    )


def _progress_label(event: ProgressEvent) -> str:
    label = PHASE_LABELS[event.phase]
    if event.total:
        label = f"{label} · {event.completed}/{event.total}"
    if event.message:
        label = f"{label} · {event.message}"
    return label


def _format_error(value: object) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


if __name__ == "__main__":
    main()
