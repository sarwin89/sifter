import io

import numpy as np
import pytest

from sifter.io import load_spectrum, preview_table


@pytest.mark.parametrize("separator", [",", "\t", ";", " "])
def test_loader_infers_supported_delimiters(separator: str) -> None:
    payload = f"energy{separator}signal\n1{separator}2\n2{separator}3\n".encode()

    preview = preview_table(io.BytesIO(payload))

    assert preview.columns == ("energy", "signal")


def test_loader_accepts_explicit_tab_after_instrument_preamble() -> None:
    payload = (
        b"Instrument export generated locally\n"
        b"wave(energy)\tintensity\n"
        b"1.50\t12.0\n"
        b"1.60\t15.5\n"
        b"1.70\t14.0\n"
        b"1.80\t18.5\n"
        b"1.90\t16.0\n"
        b"2.00\t13.5\n"
        b"2.10\t11.0\n"
        b"2.20\t9.5\n"
    )

    preview = preview_table(io.BytesIO(payload), delimiter="\t", skip_rows=1)
    spectrum = load_spectrum(
        io.BytesIO(payload),
        x_column="wave(energy)",
        intensity_column="intensity",
        delimiter="\t",
        skip_rows=1,
    )

    assert preview.columns == ("wave(energy)", "intensity")
    assert preview.delimiter == "\t"
    assert np.array_equal(spectrum.x, np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]))
    assert np.array_equal(
        spectrum.intensity,
        np.array([12.0, 15.5, 14.0, 18.5, 16.0, 13.5, 11.0, 9.5]),
    )


def test_preview_can_force_numeric_first_row_to_be_a_header() -> None:
    payload = b"100\t200\n1\t2\n2\t3\n"

    preview = preview_table(io.BytesIO(payload), delimiter="\t", header="present")

    assert preview.columns == ("100", "200")
    assert preview.rows == (("1", "2"), ("2", "3"))
    assert preview.has_header is True


def test_preview_warns_and_ignores_structurally_empty_trailing_column() -> None:
    payload = b"wave(energy)\tintensity\t\n1.50\t12.0\n1.60\t15.5\n"

    preview = preview_table(io.BytesIO(payload), delimiter="\t")

    assert preview.columns == ("wave(energy)", "intensity")
    assert preview.rows == (("1.50", "12.0"), ("1.60", "15.5"))
    assert "TRAILING_EMPTY_COLUMN_IGNORED" in preview.warnings


def test_preview_preserves_quoted_fields() -> None:
    preview = preview_table(io.BytesIO(b'x,signal,label\n1,2,"alpha, beta"\n'))

    assert preview.rows[0] == ("1", "2", "alpha, beta")


def test_headerless_preview_assigns_stable_column_names() -> None:
    preview = preview_table(io.BytesIO(b"1 2\n2 3\n3 5\n"))

    assert preview.columns == ("column_0", "column_1")
    assert preview.has_header is False
    assert "HEADER_INFERRED_ABSENT" in preview.warnings


def test_malformed_row_reports_source_line() -> None:
    with pytest.raises(ValueError, match="line 3"):
        preview_table(io.BytesIO(b"x,y\n1,2\n3,4,5\n"))


def test_load_spectrum_requires_named_columns_and_preserves_only_basename() -> None:
    rows = "\n".join(f"{index},{index**2},0.5" for index in range(8))
    source = io.BytesIO(f"energy,signal,error\n{rows}\n".encode())
    source.name = r"C:\private\experiment.csv"  # type: ignore[attr-defined]

    spectrum = load_spectrum(
        source,
        x_column="energy",
        intensity_column="signal",
        sigma_column="error",
    )

    assert np.array_equal(spectrum.x, np.arange(8.0))
    assert np.array_equal(spectrum.sigma, np.full(8, 0.5))
    assert spectrum.metadata["source_name"] == "experiment.csv"
    with pytest.raises(ValueError, match="missing column"):
        load_spectrum(
            io.BytesIO(f"energy,signal,error\n{rows}\n".encode()),
            x_column="missing",
            intensity_column="signal",
        )
