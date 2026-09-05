import pytest

from sifter import MeasurementContext


def test_measurement_context_normalizes_temperature_and_laser_power() -> None:
    context = MeasurementContext(
        temperature=25.0,
        temperature_unit="C",
        laser_power=2.5,
        laser_power_unit="mW",
        conditions={"sample": "MoS2", "spot_index": 4},
    )

    assert context.temperature_kelvin == pytest.approx(298.15)
    assert context.laser_power_watts == pytest.approx(0.0025)
    assert context.to_dict() == {
        "temperature": {"value": 25.0, "unit": "C", "kelvin": pytest.approx(298.15)},
        "laser_power": {"value": 2.5, "unit": "mW", "watts": pytest.approx(0.0025)},
        "conditions": {"sample": "MoS2", "spot_index": 4},
    }


def test_measurement_context_rejects_unphysical_or_unknown_units() -> None:
    with pytest.raises(ValueError, match="temperature"):
        MeasurementContext(temperature=-1.0, temperature_unit="K")
    with pytest.raises(ValueError, match="laser_power_unit"):
        MeasurementContext(laser_power=1.0, laser_power_unit="horsepower")
