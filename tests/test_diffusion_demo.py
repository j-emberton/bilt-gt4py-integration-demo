"""Numerical checks of the public Bilt model using independent NumPy results."""

import numpy as np
import pytest
from bilt import tensor
from bilt.units import centimeter, kelvin, meter, millimeter

from diffusion_demo import CartesianGrid2D, Diffusion, grid_implementation, implementation
from gt4py_adapter import x, y


def reference_laplacian(values: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    """Periodic NumPy reference, with values in (x, y) order."""
    x_diff = np.roll(values, 1, axis=0) + np.roll(values, -1, axis=0) - 2.0 * values
    y_diff = np.roll(values, 1, axis=1) + np.roll(values, -1, axis=1) - 2.0 * values
    return x_diff / dx**2 + y_diff / dy**2


def test_grid_coefficients() -> None:
    with grid_implementation:
        grid = CartesianGrid2D(dx=2.0 * meter, dy=4.0 * meter)
        assert grid.inv_dx2.convert_to(meter**-2).value == pytest.approx(0.25)
        assert grid.inv_dy2.convert_to(meter**-2).value == pytest.approx(0.0625)


@pytest.mark.parametrize("reverse_axes", [False, True])
@pytest.mark.parametrize("hot_spot", [(3, 2), (0, 0)])
def test_periodic_laplacian_matches_numpy(
    reverse_axes: bool,
    hot_spot: tuple[int, int],
) -> None:
    values = np.zeros((7, 5), dtype=np.float64)
    values[hot_spot] = 1.0

    # Both dimension orders must produce the same physical answer.
    ordered_values = values.T if reverse_axes else values
    dimensions = (y, x) if reverse_axes else (x, y)
    model = Diffusion(
        temperature=tensor(ordered_values, over=dimensions, unit=kelvin),
        dx=200.0 * centimeter,
        dy=1000.0 * millimeter,
    )
    with implementation:
        result = model.laplacian

    expected = reference_laplacian(values, dx=2.0, dy=1.0)
    np.testing.assert_allclose(result.values, expected.T if reverse_axes else expected)
    assert result.unit == kelvin / meter**2
    assert result.domain.dims == dimensions
    assert result.shape == ordered_values.shape
