import subprocess
from pathlib import Path


def test_public_tree_contains_no_private_or_generated_artifacts() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=Path(__file__).resolve().parents[2], text=True
    ).splitlines()
    forbidden_roots = (
        "data/",
        "private/",
        "results/",
        "outputs/",
        "reports/",
        "benchmark_results/",
    )

    assert not [path for path in tracked if path.startswith(forbidden_roots)]
    assert not [path for path in tracked if ".experimental." in path or ".fit." in path]


def test_release_documentation_covers_science_schema_and_privacy() -> None:
    root = Path(__file__).resolve().parents[2]

    assert "Fourier domain informs" in (root / "docs" / "scientific-method.md").read_text()
    assert "sifter.fit_result.v1" in (root / "docs" / "result-schema.md").read_text()
    assert "never leaves" in (root / "docs" / "privacy.md").read_text().lower()
    assert "synthetic" in (root / "CONTRIBUTING.md").read_text().lower()
