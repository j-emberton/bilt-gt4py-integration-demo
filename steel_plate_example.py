"""Compute the initial heat diffusion from a warm spot in a thin steel plate."""

import numpy as np
from bilt import Implementation, Interface, Tensor, equation, field, system, tensor

from diffusion_demo import Diffusion
from diffusion_demo import implementation as diffusion_implementation
from gt4py_adapter import x, y

GRID_POINTS = 81
DX_M = 0.006
DY_M = 0.008
AMBIENT_TEMPERATURE_C = 20.0
HOT_SPOT_RISE_C = 60.0
HOT_SPOT_WIDTH_M = 0.05
STEEL_DIFFUSIVITY_M2_PER_S = 1.2e-5
TIME_STEP_S = 0.5


@system
class HeatPlate(metaclass=Interface):
    """One explicit step of two-dimensional heat conduction."""

    temperature: Tensor = field()
    dx: float = field()
    dy: float = field()
    thermal_diffusivity: float = field()
    time_step: float = field()
    laplacian: Tensor = field()
    temperature_rate: Tensor = field()
    next_temperature: Tensor = field()


@equation
def connect_temperature(temperature: Tensor, diffusion: Diffusion) -> bool:
    """Connect the model's temperature to the numerical diffusion component."""
    return temperature == diffusion.temperature


@equation
def connect_dx(dx: float, diffusion: Diffusion) -> bool:
    """Connect x-spacing to the numerical diffusion component."""
    return dx == diffusion.dx


@equation
def connect_dy(dy: float, diffusion: Diffusion) -> bool:
    """Connect y-spacing to the numerical diffusion component."""
    return dy == diffusion.dy


@equation
def connect_laplacian(laplacian: Tensor, diffusion: Diffusion) -> bool:
    """Expose the child subsystem's Laplacian on the parent boundary."""
    return laplacian == diffusion.laplacian


@equation
def heat_equation(temperature_rate: Tensor, thermal_diffusivity: float, laplacian: Tensor) -> bool:
    """Relate the Laplacian to the temperature-change rate."""
    return temperature_rate == laplacian * thermal_diffusivity


@equation
def explicit_euler(
    next_temperature: Tensor,
    temperature: Tensor,
    temperature_rate: Tensor,
    time_step: float,
) -> bool:
    """Advance the temperature by one explicit time step."""
    return next_temperature == temperature + temperature_rate * time_step


plate_implementation = Implementation(
    HeatPlate,
    [connect_temperature, connect_dx, connect_dy, connect_laplacian, heat_equation, explicit_euler],
    internal_implementations={"diffusion": diffusion_implementation},
)


def initial_temperature() -> np.ndarray:
    """Make a smooth warm spot on a two-dimensional plate, in degrees Celsius."""
    x_positions = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DX_M
    y_positions = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DY_M
    x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing="xy")

    distance_squared = x_grid**2 + y_grid**2
    return AMBIENT_TEMPERATURE_C + HOT_SPOT_RISE_C * np.exp(-distance_squared / (2.0 * HOT_SPOT_WIDTH_M**2))


def calculate_plate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return initial temperature, its heating/cooling rate, and one Euler step."""
    temperature = initial_temperature()

    # The heat equation and timestep are Bilt equations, not NumPy post-processing.
    plate = HeatPlate(
        temperature=tensor(temperature, over=(y, x)),
        dx=DX_M,
        dy=DY_M,
        thermal_diffusivity=STEEL_DIFFUSIVITY_M2_PER_S,
        time_step=TIME_STEP_S,
    )
    with plate_implementation:
        temperature_rate = plate.temperature_rate
        next_temperature = plate.next_temperature

    return temperature, temperature_rate.values, next_temperature.values


def main() -> None:
    """Show how quickly the centre of the warm spot starts cooling."""
    temperature, rate, after_one_step = calculate_plate()
    centre = GRID_POINTS // 2
    centre_point = (centre, centre)

    print(f"Steel plate: {GRID_POINTS} x {GRID_POINTS} cells")  # noqa: T201
    print(f"Centre initially: {temperature[centre_point]:.2f} °C")  # noqa: T201
    print(f"Initial cooling rate: {rate[centre_point]:.3f} °C/s")  # noqa: T201
    print(f"Centre after {TIME_STEP_S:g} s: {after_one_step[centre_point]:.2f} °C")  # noqa: T201


if __name__ == "__main__":
    main()
