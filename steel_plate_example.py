"""Compute the initial heat diffusion from a warm spot in a thin steel plate."""

import numpy as np
from bilt import tensor

from diffusion_demo import Diffusion, implementation
from gt4py_adapter import x, y

GRID_POINTS = 81
DX_M = 0.006
DY_M = 0.008
AMBIENT_TEMPERATURE_C = 20.0
HOT_SPOT_RISE_C = 60.0
HOT_SPOT_WIDTH_M = 0.05
STEEL_DIFFUSIVITY_M2_PER_S = 1.2e-5
TIME_STEP_S = 0.5


def initial_temperature() -> np.ndarray:
    """Make a smooth warm spot on a two-dimensional plate, in degrees Celsius."""
    x_positions = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DX_M
    y_positions = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DY_M
    x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing="xy")

    distance_squared = x_grid**2 + y_grid**2
    return AMBIENT_TEMPERATURE_C + HOT_SPOT_RISE_C * np.exp(
        -distance_squared / (2.0 * HOT_SPOT_WIDTH_M**2)
    )


def calculate_plate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial temperature, its heating/cooling rate, and one Euler step."""
    temperature = initial_temperature()

    # Bilt owns the spatial field and the spacing; GT4Py evaluates the Laplacian.
    model = Diffusion(
        temperature=tensor(temperature, over=(y, x)),
        dx=DX_M,
        dy=DY_M,
    )
    with implementation:
        laplacian = model.laplacian

    # Heat equation: dT/dt = alpha * laplacian(T).
    rate_c_per_s = STEEL_DIFFUSIVITY_M2_PER_S * laplacian.values
    after_one_step = temperature + TIME_STEP_S * rate_c_per_s
    return temperature, rate_c_per_s, after_one_step


def main() -> None:
    """Show how quickly the centre of the warm spot starts cooling."""
    temperature, rate, after_one_step = calculate_plate()
    centre = GRID_POINTS // 2
    centre_point = (centre, centre)

    print(f"Steel plate: {GRID_POINTS} × {GRID_POINTS} cells")  # noqa: T201
    print(f"Centre initially: {temperature[centre_point]:.2f} °C")  # noqa: T201
    print(f"Initial cooling rate: {rate[centre_point]:.3f} °C/s")  # noqa: T201
    print(f"Centre after {TIME_STEP_S:g} s: {after_one_step[centre_point]:.2f} °C")  # noqa: T201


if __name__ == "__main__":
    main()
