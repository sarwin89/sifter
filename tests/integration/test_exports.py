import json
from dataclasses import replace

from sifter import AutofitConfig, MeasurementContext, autofit
from tests.helpers import easy_one_peak_spectrum


def test_dataframe_is_a_flat_peak_table() -> None:
    result = _fitted_result()

    table = result.to_dataframe()

    assert list(table["peak_index"]) == [0]
    assert {"shape", "area", "center", "sigma", "gamma", "bic", "aicc"} <= set(table.columns)


def test_json_is_standard_compliant_and_excludes_private_paths() -> None:
    result = _fitted_result()
    unavailable_model = replace(result.best_model, rmse=float("nan"))
    private_root = r"C:\private\instrument"
    private = replace(
        result,
        source_metadata={"source_path": private_root + r"\secret.txt"},
        best_model=unavailable_model,
    )

    encoded = private.to_json()
    decoded = json.loads(encoded, parse_constant=_reject_nonstandard_constant)

    assert decoded["schema_version"] == "sifter.fit_result.v2"
    assert decoded["settings"]["search_mode"] == "standard"
    assert decoded["measurement_context"] is None
    assert decoded["best_model"]["rmse"] is None
    assert "NONFINITE_VALUE_OMITTED" in {warning["code"] for warning in decoded["warnings"]}
    assert private_root not in encoded
    assert decoded["source_metadata"]["source_path"] == "secret.txt"
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_json_export_is_deterministic() -> None:
    result = _fitted_result()

    assert result.to_json() == result.to_json()


def test_legacy_v1_result_json_remains_supported() -> None:
    legacy = replace(_fitted_result(), schema_version="sifter.fit_result.v1")

    decoded = json.loads(legacy.to_json(), parse_constant=_reject_nonstandard_constant)

    assert decoded["schema_version"] == "sifter.fit_result.v1"
    assert "measurement_context" not in decoded


def test_v2_json_records_measurement_context_as_provenance() -> None:
    result = autofit(
        easy_one_peak_spectrum(),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
            measurement_context=MeasurementContext(
                temperature=300.0,
                temperature_unit="K",
                laser_power=1.2,
                laser_power_unit="uW",
            ),
        ),
    )

    decoded = json.loads(result.to_json(), parse_constant=_reject_nonstandard_constant)

    assert decoded["schema_version"] == "sifter.fit_result.v2"
    assert decoded["measurement_context"]["temperature"]["kelvin"] == 300.0
    assert decoded["measurement_context"]["laser_power"]["watts"] == 1.2e-6


def _fitted_result():  # type: ignore[no-untyped-def]
    return autofit(
        easy_one_peak_spectrum(),
        config=AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
        ),
    )


def _reject_nonstandard_constant(value: str) -> None:
    raise AssertionError(f"non-standard JSON constant: {value}")
