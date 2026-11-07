"""Compare the steel-plate example against an independent finite-difference result."""

import numpy as np
import pytest

from steel_plate_example import (
    DX_M,
    DY_M,
    GRID_POINTS,
    STEEL_DIFFUSIVITY_M2_PER_S,
    TIME_STEP_S,
    calculate_plate,
)


def test_initial_steel_plate_heat_flow() -> None:
    temperature, rate_c_per_s, updated = calculate_plate()

    # The model stores the array in (y, x) order; x therefore uses axis 1.
    x_curvature = (
        np.roll(temperature, 1, axis=1)
        - 2.0 * temperature
        + np.roll(temperature, -1, axis=1)
    ) / DX_M**2
    y_curvature = (
        np.roll(temperature, 1, axis=0)
        - 2.0 * temperature
        + np.roll(temperature, -1, axis=0)
    ) / DY_M**2
    expected_rate = STEEL_DIFFUSIVITY_M2_PER_S * (x_curvature + y_curvature)

    assert temperature.shape == (GRID_POINTS, GRID_POINTS)
    np.testing.assert_allclose(rate_c_per_s, expected_rate, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(updated, temperature + TIME_STEP_S * expected_rate, rtol=1e-10, atol=1e-10)

    centre = GRID_POINTS // 2
    assert rate_c_per_s[centre, centre] < 0  # The hot centre cools.
    assert np.max(rate_c_per_s) > 0  # Cooler neighbours warm up.
    assert np.count_nonzero(np.abs(rate_c_per_s) > 1e-4) > GRID_POINTS

    # A periodic heat equation has no net source: the mean temperature is conserved.
    assert np.mean(updated) == pytest.approx(np.mean(temperature), abs=1e-9)
