from importlib.metadata import version


def test_distribution_and_namespace_are_importable() -> None:
    import sifter

    assert version("sifter") == sifter.__version__
    assert sifter.__version__ == "0.2.1"
