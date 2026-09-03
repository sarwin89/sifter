"""Plotly figures assembled exclusively from immutable FitResult data."""

from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

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


def render_fit_png(result: FitResult, *, width: int = 1600, height: int = 900) -> bytes:
    """Render a dependency-light publication-neutral fit preview as PNG."""
    if width < 640 or height < 360:
        raise ValueError("PNG dimensions must be at least 640 by 360")
    image = Image.new("RGB", (width, height), "#f5f2e9")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 110, 70, width - 60, height - 110
    draw.rectangle((left, top, right, bottom), fill="#fbfaf5", outline="#29483d", width=2)
    x_min, x_max = float(result.x[0]), float(result.x[-1])
    combined = np.concatenate((result.intensity, result.best_model.fitted))
    y_min, y_max = float(np.min(combined)), float(np.max(combined))
    y_span = max(y_max - y_min, np.finfo(float).eps)

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        return [
            (
                left + (float(x_value) - x_min) / (x_max - x_min) * (right - left),
                bottom - (float(y_value) - y_min) / y_span * (bottom - top),
            )
            for x_value, y_value in zip(result.x, values, strict=True)
        ]

    draw.line(points(result.intensity), fill="#66746e", width=2)
    for component in result.best_model.components:
        draw.line(points(component + result.best_model.baseline), fill="#ba8d3b", width=2)
    draw.line(points(result.best_model.fitted), fill="#176b55", width=4)
    draw.text((left, 26), "SIFTER — recommended spectral decomposition", fill="#17231f")
    draw.text(
        (left, bottom + 34),
        f"{result.best_model.peak_count} {result.best_model.shape} peaks  |  "
        f"BIC {result.best_model.bic:.3f}  |  RMSE {result.best_model.rmse:.4g}",
        fill="#29483d",
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _axis_title(name: str, unit: str | None) -> str:
    return name if unit is None else f"{name} ({unit})"
