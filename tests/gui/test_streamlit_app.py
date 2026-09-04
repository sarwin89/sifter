from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from sifter.synthetic import SyntheticPeak, make_spectrum

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


def test_initial_view_teaches_local_workflow_without_running_fit() -> None:
    app = AppTest.from_file(APP_PATH).run()

    assert app.exception == []
    assert len(app.file_uploader) == 1
    assert any("Your spectrum stays on this machine" in item.value for item in app.info)
    assert not any("Recommended model" in item.value for item in app.subheader)


def test_upload_exposes_confirmable_defaults_but_waits_for_analyze() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].upload("synthetic.csv", _two_peak_csv(), "text/csv").run()

    assert app.exception == []
    assert app.selectbox(key="x_column").value == "x"
    assert app.selectbox(key="intensity_column").value == "intensity"
    assert app.number_input(key="max_peaks").value == 10
    assert app.number_input(key="max_peaks").max == 10
    assert app.multiselect(key="shapes").value == ["Gaussian", "Lorentzian", "Voigt"]
    assert app.checkbox(key="fourier_enabled").value is True
    assert app.number_input(key="random_seed").value == 42
    assert not any("Recommended model" in item.value for item in app.subheader)


def test_import_controls_recover_tab_table_after_instrument_preamble() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].upload(
        "instrument.txt",
        _tab_spectrum_with_preamble(),
        "text/plain",
    ).run()

    app.selectbox(key="input_delimiter").select("Tab")
    app.number_input(key="input_skip_rows").set_value(1)
    app.run()

    assert app.exception == []
    assert app.selectbox(key="x_column").value == "wave(energy)"
    assert app.selectbox(key="intensity_column").value == "intensity"
    assert not app.error


def test_synthetic_upload_reaches_results_view_and_exports() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.file_uploader[0].upload("synthetic.csv", _two_peak_csv(), "text/csv").run()
    app.number_input(key="max_peaks").set_value(2)
    app.multiselect(key="shapes").set_value(["Gaussian"])
    app.multiselect(key="baselines").set_value([0])
    app.button(key="analyze").click().run(timeout=60)

    assert app.exception == []
    assert any("Recommended model" in item.value for item in app.subheader)
    assert len(app.get("plotly_chart")) >= 2
    assert len(app.download_button) == 3
    assert any("Candidate comparison" in item.value for item in app.subheader)
    assert any("Covariance" in item.value for item in app.caption)


def _two_peak_csv() -> bytes:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 3.0, 241),
        peaks=(
            SyntheticPeak("gaussian", area=2.0, center=1.0, sigma=0.08),
            SyntheticPeak("gaussian", area=1.5, center=1.7, sigma=0.09),
        ),
        baseline=(0.2,),
        noise="gaussian",
        snr=120.0,
        seed=7,
    )
    rows = ["x,intensity"]
    rows.extend(
        f"{x_value:.12g},{intensity:.12g}"
        for x_value, intensity in zip(spectrum.x, spectrum.intensity, strict=True)
    )
    return ("\n".join(rows) + "\n").encode()


def _tab_spectrum_with_preamble() -> bytes:
    rows = [
        "Instrument export generated locally",
        "wave(energy)\tintensity",
        "1.50\t12.0",
        "1.60\t15.5",
        "1.70\t14.0",
        "1.80\t18.5",
        "1.90\t16.0",
        "2.00\t13.5",
        "2.10\t11.0",
        "2.20\t9.5",
    ]
    return ("\n".join(rows) + "\n").encode()
