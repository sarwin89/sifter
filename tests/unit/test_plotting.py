from sifter import AutofitConfig, autofit
from sifter.plotting import render_fit_png
from tests.helpers import easy_one_peak_spectrum


def test_plot_contains_data_fit_components_residuals_and_fourier() -> None:
    result = autofit(
        easy_one_peak_spectrum(),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=True,
        ),
    )

    figures = result.plot()

    assert set(figures) == {"fit", "residuals", "fourier"}
    assert {trace.name for trace in figures["fit"].data} >= {
        "Observed",
        "Recommended fit",
        "Baseline",
        "Peak 1",
    }
    assert {trace.name for trace in figures["residuals"].data} == {"Residuals"}
    assert {trace.name for trace in figures["fourier"].data} >= {"Fourier magnitude"}
    assert all(figure.layout.paper_bgcolor == "#fbfaf5" for figure in figures.values())
    assert all(figure.layout.font.color == "#17231f" for figure in figures.values())
    assert render_fit_png(result).startswith(b"\x89PNG\r\n\x1a\n")
