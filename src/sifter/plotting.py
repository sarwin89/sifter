"""Plotly figures assembled exclusively from immutable FitResult data."""

import plotly.graph_objects as go

from sifter.result import FitResult


def plot_result(result: FitResult) -> dict[str, go.Figure]:
    """Build decomposition, residual, and optional Fourier figures."""
    fit_figure = go.Figure()
    fit_figure.add_scatter(x=result.x, y=result.intensity, mode="markers", name="Observed")
    fit_figure.add_scatter(
        x=result.x,
        y=result.best_model.fitted,
        mode="lines",
        name="Recommended fit",
    )
    fit_figure.add_scatter(
        x=result.x,
        y=result.best_model.baseline,
        mode="lines",
        name="Baseline",
    )
    for index, component in enumerate(result.best_model.components, start=1):
        fit_figure.add_scatter(x=result.x, y=component, mode="lines", name=f"Peak {index}")
    fit_figure.update_layout(
        xaxis_title=_axis_title(result.x_name, result.x_unit),
        yaxis_title=result.intensity_name,
        template="plotly_white",
    )

    residual_figure = go.Figure()
    residual_figure.add_scatter(
        x=result.x,
        y=result.best_model.residuals,
        mode="markers",
        name="Residuals",
    )
    residual_figure.add_hline(y=0.0, line_dash="dash")
    residual_figure.update_layout(
        xaxis_title=_axis_title(result.x_name, result.x_unit),
        yaxis_title="fit - observed",
        template="plotly_white",
    )
    figures = {"fit": fit_figure, "residuals": residual_figure}
    if result.fourier is not None:
        fourier_figure = go.Figure()
        fourier_figure.add_scatter(
            x=result.fourier.frequency,
            y=result.fourier.magnitude,
            mode="lines",
            name="Fourier magnitude",
        )
        fourier_figure.update_layout(
            xaxis_title="frequency",
            yaxis_title="magnitude",
            template="plotly_white",
        )
        figures["fourier"] = fourier_figure
    return figures


def _axis_title(name: str, unit: str | None) -> str:
    return name if unit is None else f"{name} ({unit})"
