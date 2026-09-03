import io

import numpy as np
import pytest

from sifter.io import load_spectrum, preview_table


@pytest.mark.parametrize("separator", [",", "\t", ";", " "])
def test_loader_infers_supported_delimiters(separator: str) -> None:
    payload = f"energy{separator}signal\n1{separator}2\n2{separator}3\n".encode()

    preview = preview_table(io.BytesIO(payload))

    assert preview.columns == ("energy", "signal")


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
