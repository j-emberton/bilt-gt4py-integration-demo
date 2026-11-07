"""Demonstrate a GT4Py stencil evaluated through an ordinary Bilt subsystem."""

from typing import Annotated

import numpy as np
from bilt import Implementation, Interface, Quantity, Tensor, equation, field, system, tensor, transform
from bilt.units import kelvin, meter
from bilt.units.dimensions import LENGTH, TEMPERATURE

from gt4py_adapter import x, y

Length = Annotated[Quantity, LENGTH]
InverseArea = Annotated[Quantity, LENGTH**-2]
TemperatureDifference = Annotated[Tensor, TEMPERATURE]
TemperatureLaplacian = Annotated[Tensor, TEMPERATURE / LENGTH**2]


@system
class CartesianGrid2D(metaclass=Interface):
    """Uniform Cartesian grid spacing and its derived Laplacian coefficients."""

    dx: Length = field()
    dy: Length = field()
    inv_dx2: InverseArea = field()
    inv_dy2: InverseArea = field()


@transform
def inv_dx2(dx: Length) -> InverseArea:
    return 1.0 / dx**2


@transform
def inv_dy2(dy: Length) -> InverseArea:
    return 1.0 / dy**2


grid_implementation = Implementation(CartesianGrid2D, [inv_dx2, inv_dy2])


@system
class Laplacian2D(metaclass=Interface):
    """A two-dimensional spatial Laplacian of a relative temperature field."""

    temperature: TemperatureDifference = field()
    dx: Length = field()
    dy: Length = field()
    laplacian: TemperatureLaplacian = field()


@equation
def connect_dx(dx: Length, grid: CartesianGrid2D) -> bool:
    return dx == grid.dx


@equation
def connect_dy(dy: Length, grid: CartesianGrid2D) -> bool:
    return dy == grid.dy


@equation
def connect_inverse_dx_squared(inv_dx2: InverseArea, grid: CartesianGrid2D) -> bool:
    return inv_dx2 == grid.inv_dx2


@equation
def connect_inverse_dy_squared(inv_dy2: InverseArea, grid: CartesianGrid2D) -> bool:
    return inv_dy2 == grid.inv_dy2


@transform
def laplacian(temperature: TemperatureDifference, inv_dx2: InverseArea, inv_dy2: InverseArea) -> TemperatureLaplacian:
    # Imports stay inside the transform because Bilt compiles its function AST.
    from gt4py_adapter import periodic_laplacian  # noqa: PLC0415

    # Bilt lowers Tensor arguments/results to NumPy arrays for Python execution.
    return periodic_laplacian(temperature, inv_dx2, inv_dy2)  # pyrefly: ignore[bad-argument-type, bad-return]  # ty: ignore[invalid-argument-type, invalid-return-type]


implementation = Implementation(
    Laplacian2D,
    [
        connect_dx,
        connect_dy,
        connect_inverse_dx_squared,
        connect_inverse_dy_squared,
        laplacian,
    ],
    internal_implementations={"grid": grid_implementation},
)


def main() -> None:
    """Run a periodic five-point Laplacian with a reversed Bilt axis order."""
    values = np.zeros((5, 7), dtype=np.float64)
    values[2, 3] = 1.0

    # The model's (y, x) ordering differs from GT4Py's (I, J) ordering.
    model = Laplacian2D(temperature=tensor(values, over=(y, x), unit=kelvin), dx=2.0 * meter, dy=1.0 * meter)
    with implementation:
        result = model.laplacian

    expected = np.zeros_like(values)
    expected[2, 3] = -2.5
    expected[2, 2] = expected[2, 4] = 0.25
    expected[1, 3] = expected[3, 3] = 1.0
    np.testing.assert_allclose(result.values, expected)
    print("Bilt + GT4Py: periodic Laplacian verified")  # noqa: T201


if __name__ == "__main__":
    main()
