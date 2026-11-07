"""Demonstrate a GT4Py stencil evaluated through an ordinary Bilt subsystem."""

import numpy as np
from bilt import Implementation, Interface, Tensor, equation, field, system, tensor, transform

from gt4py_adapter import x, y


@system
class CartesianGrid2D(metaclass=Interface):
    """Uniform Cartesian grid spacing and its derived Laplacian coefficients."""

    dx: float = field()
    dy: float = field()
    inv_dx2: float = field()
    inv_dy2: float = field()


@transform
def inv_dx2(dx: float) -> float:
    return 1.0 / dx**2


@transform
def inv_dy2(dy: float) -> float:
    return 1.0 / dy**2


grid_implementation = Implementation(CartesianGrid2D, [inv_dx2, inv_dy2])


@system
class Diffusion(metaclass=Interface):
    """Temperature field and its discrete two-dimensional Laplacian."""

    temperature: Tensor = field()
    dx: float = field()
    dy: float = field()
    laplacian: Tensor = field()


@equation
def connect_dx(dx: float, grid: CartesianGrid2D) -> bool:
    return dx == grid.dx


@equation
def connect_dy(dy: float, grid: CartesianGrid2D) -> bool:
    return dy == grid.dy


@equation
def connect_inverse_dx_squared(inv_dx2: float, grid: CartesianGrid2D) -> bool:
    return inv_dx2 == grid.inv_dx2


@equation
def connect_inverse_dy_squared(inv_dy2: float, grid: CartesianGrid2D) -> bool:
    return inv_dy2 == grid.inv_dy2


@transform
def laplacian(temperature: Tensor, inv_dx2: float, inv_dy2: float) -> Tensor:
    # Imports stay inside the transform because Bilt compiles its function AST.
    from gt4py_adapter import periodic_laplacian  # noqa: PLC0415

    # Bilt lowers Tensor arguments/results to NumPy arrays for Python execution.
    return periodic_laplacian(temperature, inv_dx2, inv_dy2)  # pyrefly: ignore[bad-argument-type, bad-return]  # ty: ignore[invalid-argument-type, invalid-return-type]


implementation = Implementation(
    Diffusion,
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
    model = Diffusion(temperature=tensor(values, over=(y, x)), dx=2.0, dy=1.0)
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
