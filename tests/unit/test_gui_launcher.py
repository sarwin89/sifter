import sys

import pytest

from sifter.gui import main


def test_launcher_invokes_local_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def record(args: list[str], *, check: bool) -> None:
        calls.append((args, check))

    monkeypatch.setattr("sifter.gui.subprocess.run", record)

    main()

    args, check = calls[0]
    assert args[:3] == [sys.executable, "-m", "streamlit"]
    assert args[3] == "run"
    assert args[-2:] == ["--server.address", "localhost"]
    assert check is True
