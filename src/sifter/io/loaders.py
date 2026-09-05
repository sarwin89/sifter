"""Local delimited-table preview and Spectrum loading."""

import csv
import io
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Literal, TypeAlias, cast

import numpy as np

from sifter.spectrum import Spectrum

Source: TypeAlias = bytes | bytearray | str | PathLike[str] | BinaryIO
Delimiter: TypeAlias = Literal[",", "\t", ";", "whitespace"]
DelimiterOption: TypeAlias = Delimiter | Literal["auto"]
HeaderMode: TypeAlias = Literal["auto", "present", "absent"]
PREVIEW_LIMIT = 64 * 1024


@dataclass(frozen=True, slots=True)
class TablePreview:
    """Inferred structure and rows from a local delimited table."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    delimiter: Delimiter
    has_header: bool
    warnings: tuple[str, ...]


def preview_table(
    source: Source,
    *,
    delimiter: DelimiterOption = "auto",
    header: HeaderMode = "auto",
    skip_rows: int = 0,
) -> TablePreview:
    """Read at most 64 KiB and infer a supported delimiter and header."""
    payload = _read_source(source, limit=PREVIEW_LIMIT)
    return _parse_preview(payload, delimiter=delimiter, header=header, skip_rows=skip_rows)


def load_spectrum(
    source: Source,
    *,
    x_column: str,
    intensity_column: str,
    sigma_column: str | None = None,
    x_name: str | None = None,
    x_unit: str | None = None,
    intensity_name: str | None = None,
    delimiter: DelimiterOption = "auto",
    header: HeaderMode = "auto",
    skip_rows: int = 0,
) -> Spectrum:
    """Load explicitly selected numeric columns from a local table."""
    payload = _read_source(source)
    preview = _parse_preview(
        payload,
        delimiter=delimiter,
        header=header,
        skip_rows=skip_rows,
    )
    required = (x_column, intensity_column) + (() if sigma_column is None else (sigma_column,))
    missing = [name for name in required if name not in preview.columns]
    if missing:
        raise ValueError(f"missing column: {missing[0]}")
    positions = {name: preview.columns.index(name) for name in required}
    values: dict[str, list[float]] = {name: [] for name in required}
    first_data_line = skip_rows + (2 if preview.has_header else 1)
    for row_index, row in enumerate(preview.rows, start=first_data_line):
        for name, position in positions.items():
            try:
                values[name].append(float(row[position]))
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"column {name!r} contains a nonnumeric value on line {row_index}"
                ) from error
    metadata: dict[str, str] = {}
    source_name = _source_name(source)
    if source_name is not None:
        metadata["source_name"] = source_name
    return Spectrum(
        values[x_column],
        values[intensity_column],
        sigma=None if sigma_column is None else values[sigma_column],
        x_name=x_column if x_name is None else x_name,
        x_unit=x_unit,
        intensity_name=intensity_column if intensity_name is None else intensity_name,
        metadata=metadata,
    )


def _parse_preview(
    payload: bytes,
    *,
    delimiter: DelimiterOption = "auto",
    header: HeaderMode = "auto",
    skip_rows: int = 0,
) -> TablePreview:
    if isinstance(skip_rows, bool) or skip_rows < 0:
        raise ValueError("skip_rows must be a nonnegative integer")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("table must be UTF-8 encoded") from error
    lines = [line for line in text.splitlines()[skip_rows:] if line.strip()]
    if not lines:
        raise ValueError("table is empty")
    selected_delimiter = (
        _infer_delimiter("\n".join(lines[:20])) if delimiter == "auto" else delimiter
    )
    rows = _read_rows(lines, selected_delimiter)
    rows, ignored_trailing_column = _ignore_structurally_empty_trailing_columns(rows)
    expected = len(rows[0])
    if expected < 2:
        raise ValueError("table must contain at least two columns")
    for line_number, row in enumerate(rows, start=skip_rows + 1):
        if len(row) != expected:
            raise ValueError(
                f"malformed row on line {line_number}: expected {expected} columns, "
                f"found {len(row)}"
            )
    has_header = (
        not all(_is_float(cell) for cell in rows[0])
        if header == "auto"
        else header == "present"
    )
    if has_header:
        columns = tuple(cell.strip() for cell in rows[0])
        data_rows = rows[1:]
        warnings: tuple[str, ...] = (
            ("TRAILING_EMPTY_COLUMN_IGNORED",) if ignored_trailing_column else ()
        )
    else:
        columns = tuple(f"column_{index}" for index in range(expected))
        data_rows = rows
        warnings = ("HEADER_INFERRED_ABSENT",) + (
            ("TRAILING_EMPTY_COLUMN_IGNORED",) if ignored_trailing_column else ()
        )
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("header columns must be nonempty and unique")
    return TablePreview(
        columns=columns,
        rows=tuple(tuple(cell for cell in row) for row in data_rows),
        delimiter=selected_delimiter,
        has_header=has_header,
        warnings=warnings,
    )


def _infer_delimiter(sample: str) -> Delimiter:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        character = dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        counts = {candidate: first_line.count(candidate) for candidate in (",", "\t", ";", " ")}
        character = max(counts, key=counts.__getitem__)
        if counts[character] == 0:
            raise ValueError("could not infer a supported delimiter") from None
    if character == " ":
        return "whitespace"
    if character not in {",", "\t", ";"}:
        raise ValueError("unsupported delimiter")
    return cast(Delimiter, character)


def _read_rows(lines: list[str], delimiter: Delimiter) -> list[list[str]]:
    character = " " if delimiter == "whitespace" else delimiter
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=character, skipinitialspace=True)
    return [[cell.strip() for cell in row if delimiter != "whitespace" or cell] for row in reader]


def _ignore_structurally_empty_trailing_columns(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    meaningful_widths = [
        next((index + 1 for index in range(len(row) - 1, -1, -1) if row[index]), 0)
        for row in rows
    ]
    if not meaningful_widths or len(set(meaningful_widths)) != 1:
        return rows, False
    width = meaningful_widths[0]
    ignored = any(len(row) > width for row in rows)
    return ([row[:width] for row in rows] if ignored else rows), ignored


def _read_source(source: Source, *, limit: int | None = None) -> bytes:
    if isinstance(source, bytes):
        return source if limit is None else source[:limit]
    if isinstance(source, bytearray):
        payload = bytes(source)
        return payload if limit is None else payload[:limit]
    if isinstance(source, (str, PathLike)):
        with Path(source).open("rb") as handle:
            return handle.read() if limit is None else handle.read(limit)
    position = source.tell() if source.seekable() else None
    payload = source.read() if limit is None else source.read(limit)
    if position is not None:
        source.seek(position)
    if not isinstance(payload, bytes):
        raise TypeError("source must provide bytes")
    return payload


def _source_name(source: Source) -> str | None:
    if isinstance(source, (str, PathLike)):
        return Path(source).name
    name = getattr(source, "name", None)
    if not isinstance(name, str):
        return None
    return PureWindowsPath(name).name


def _is_float(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return bool(np.isfinite(parsed))
