"""Compute the initial heat diffusion from a warm spot in a thin steel plate."""

from typing import Annotated

import numpy as np
from bilt import Implementation, Interface, Quantity, Tensor, equation, field, system, tensor
from bilt.units import absolute_celsius, meter, second
from bilt.units.dimensions import ABSOLUTE, LENGTH, TEMPERATURE, TIME

from diffusion_demo import Diffusion, Length, TemperatureDifference, TemperatureLaplacian
from diffusion_demo import implementation as diffusion_implementation
from gt4py_adapter import x, y

AbsoluteTemperature = Annotated[Quantity, TEMPERATURE, ABSOLUTE]
AbsoluteTemperatureField = Annotated[Tensor, TEMPERATURE, ABSOLUTE]
ThermalDiffusivity = Annotated[Quantity, LENGTH**2 / TIME]
Duration = Annotated[Quantity, TIME]
TemperatureRate = Annotated[Tensor, TEMPERATURE / TIME]

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

    temperature: AbsoluteTemperatureField = field()
    ambient_temperature: AbsoluteTemperature = field()
    temperature_difference: TemperatureDifference = field()
    dx: Length = field()
    dy: Length = field()
    thermal_diffusivity: ThermalDiffusivity = field()
    time_step: Duration = field()
    laplacian: TemperatureLaplacian = field()
    temperature_rate: TemperatureRate = field()
    next_temperature: AbsoluteTemperatureField = field()


@equation
def ambient_difference(
    temperature_difference: TemperatureDifference,
    temperature: AbsoluteTemperatureField,
    ambient_temperature: AbsoluteTemperature,
) -> bool:
    """Express the field as a temperature change relative to ambient."""
    return temperature_difference == temperature - ambient_temperature


@equation
def connect_temperature(temperature_difference: TemperatureDifference, diffusion: Diffusion) -> bool:
    """Connect the relative temperature field to the diffusion subsystem."""
    return temperature_difference == diffusion.temperature


@equation
def connect_dx(dx: Length, diffusion: Diffusion) -> bool:
    """Connect x-spacing to the numerical diffusion component."""
    return dx == diffusion.dx


@equation
def connect_dy(dy: Length, diffusion: Diffusion) -> bool:
    """Connect y-spacing to the numerical diffusion component."""
    return dy == diffusion.dy


@equation
def connect_laplacian(laplacian: TemperatureLaplacian, diffusion: Diffusion) -> bool:
    """Expose the child subsystem's Laplacian on the parent boundary."""
    return laplacian == diffusion.laplacian


@equation
def heat_equation(
    temperature_rate: TemperatureRate,
    thermal_diffusivity: ThermalDiffusivity,
    laplacian: TemperatureLaplacian,
) -> bool:
    """Relate the Laplacian to the temperature-change rate."""
    return temperature_rate == laplacian * thermal_diffusivity


@equation
def explicit_euler(
    next_temperature: AbsoluteTemperatureField,
    temperature: AbsoluteTemperatureField,
    temperature_rate: TemperatureRate,
    time_step: Duration,
) -> bool:
    """Advance the temperature by one explicit time step."""
    return next_temperature == temperature + temperature_rate * time_step


plate_implementation = Implementation(
    HeatPlate,
    [ambient_difference, connect_temperature, connect_dx, connect_dy, connect_laplacian, heat_equation, explicit_euler],
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
        temperature=tensor(temperature, over=(y, x), unit=absolute_celsius),
        ambient_temperature=Quantity(AMBIENT_TEMPERATURE_C, absolute_celsius),
        dx=DX_M * meter,
        dy=DY_M * meter,
        thermal_diffusivity=STEEL_DIFFUSIVITY_M2_PER_S * meter**2 / second,
        time_step=TIME_STEP_S * second,
    )
    with plate_implementation:
        temperature_rate = plate.temperature_rate
        next_temperature = plate.next_temperature

    return temperature, temperature_rate.values, next_temperature.convert_to(absolute_celsius).values


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
