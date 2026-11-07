"""Compare the steel-plate example against an independent finite-difference result."""

import numpy as np
import pytest
from bilt import Quantity, tensor
from bilt.units import absolute_celsius, absolute_kelvin, kelvin, meter, second

from gt4py_adapter import x, y
from steel_plate_example import (
    AMBIENT_TEMPERATURE_C,
    DX_M,
    DY_M,
    GRID_POINTS,
    STEEL_DIFFUSIVITY_M2_PER_S,
    TIME_STEP_S,
    HeatPlate,
    calculate_plate,
    initial_temperature,
    plate_implementation,
)


def test_initial_steel_plate_heat_flow() -> None:
    temperature, rate_c_per_s, updated = calculate_plate()

    # The model stores the array in (y, x) order; x therefore uses axis 1.
    x_curvature = (np.roll(temperature, 1, axis=1) - 2.0 * temperature + np.roll(temperature, -1, axis=1)) / DX_M**2
    y_curvature = (np.roll(temperature, 1, axis=0) - 2.0 * temperature + np.roll(temperature, -1, axis=0)) / DY_M**2
    expected_rate = STEEL_DIFFUSIVITY_M2_PER_S * (x_curvature + y_curvature)

    assert temperature.shape == (GRID_POINTS, GRID_POINTS)
    np.testing.assert_allclose(rate_c_per_s, expected_rate, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(updated, temperature + TIME_STEP_S * expected_rate, rtol=1e-10, atol=1e-10)

    centre = GRID_POINTS // 2
    assert rate_c_per_s[centre, centre] < 0  # The hot centre cools.
    assert np.max(rate_c_per_s) > 0  # Cooler neighbours warm up.
    measurable_rate_c_per_s = 1e-4
    assert np.count_nonzero(np.abs(rate_c_per_s) > measurable_rate_c_per_s) > GRID_POINTS

    # A periodic heat equation has no net source: the mean temperature is conserved.
    assert np.mean(updated) == pytest.approx(np.mean(temperature), abs=1e-9)


def test_heat_equation_and_step_are_model_relationships() -> None:
    temperature = initial_temperature()
    thermal_diffusivity = 0.5 * STEEL_DIFFUSIVITY_M2_PER_S
    time_step = 0.25
    model = HeatPlate(
        temperature=tensor(temperature, over=(y, x), unit=absolute_celsius),
        ambient_temperature=Quantity(AMBIENT_TEMPERATURE_C, absolute_celsius),
        dx=DX_M * meter,
        dy=DY_M * meter,
        thermal_diffusivity=thermal_diffusivity * meter**2 / second,
        time_step=time_step * second,
    )

    with plate_implementation:
        laplacian = model.laplacian
        temperature_rate = model.temperature_rate
        next_temperature = model.next_temperature

    x_curvature = (np.roll(temperature, 1, axis=1) - 2.0 * temperature + np.roll(temperature, -1, axis=1)) / DX_M**2
    y_curvature = (np.roll(temperature, 1, axis=0) - 2.0 * temperature + np.roll(temperature, -1, axis=0)) / DY_M**2
    expected_laplacian = x_curvature + y_curvature
    expected_rate = thermal_diffusivity * expected_laplacian

    assert laplacian.unit == kelvin / meter**2
    # Subtracting absolute kelvin readings amplifies roundoff in the second derivative.
    np.testing.assert_allclose(laplacian.values, expected_laplacian, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(temperature_rate.values, expected_rate, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        next_temperature.convert_to(absolute_celsius).values,
        temperature + time_step * expected_rate,
        rtol=1e-10,
        atol=1e-10,
    )
    assert temperature_rate.unit == kelvin / second
    assert next_temperature.unit == absolute_kelvin
    assert laplacian.domain.dims == (y, x)
    assert temperature_rate.domain.dims == (y, x)
    assert next_temperature.domain.dims == (y, x)
