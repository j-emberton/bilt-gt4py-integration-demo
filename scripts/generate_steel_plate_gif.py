"""Regenerate steel-plate animation using repeated Bilt/GT4Py heat-equation steps.

Run `uv run python -m scripts.generate_steel_plate_gif` (`--check` to verify).
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np  # noqa: TC002
from bilt import Quantity, tensor
from bilt.units import absolute_celsius, meter, second

from gt4py_adapter import x, y
from steel_plate_example import (
    AMBIENT_TEMPERATURE_C,
    DX_M,
    DY_M,
    GRID_POINTS,
    HOT_SPOT_RISE_C,
    STEEL_DIFFUSIVITY_M2_PER_S,
    TIME_STEP_S,
    HeatPlate,
    initial_temperature,
    plate_implementation,
)

WIDTH, HEIGHT = 312, 396
FRAME_COUNT = 9
STEPS_PER_FRAME = 40
BITS_PER_BYTE = 8
MAX_CODES = 4096
MAX_CODE_WIDTH = 12
PALETTE_STOPS = ((19, 36, 83), (27, 101, 163), (45, 181, 191), (248, 212, 84), (190, 50, 44))
GLYPHS: dict[str, str] = dict(
    zip(
        "0123456789TS=CHEAPL ",
        [
            "01110 10001 10011 10101 11001 10001 01110",
            "00100 01100 00100 00100 00100 00100 01110",
            "01110 10001 00001 00010 00100 01000 11111",
            "11110 00001 00001 01110 00001 00001 11110",
            "00010 00110 01010 10010 11111 00010 00010",
            "11111 10000 10000 11110 00001 00001 11110",
            "00111 01000 10000 11110 10001 10001 01110",
            "11111 00001 00010 00100 01000 01000 01000",
            "01110 10001 10001 01110 10001 10001 01110",
            "01110 10001 10001 01111 00001 00010 11100",
            "11111 00100 00100 00100 00100 00100 00100",
            "01111 10000 10000 01110 00001 00001 11110",
            "00000 00000 11111 00000 11111 00000 00000",
            "01111 10000 10000 10000 10000 10000 01111",
            "10001 10001 10001 11111 10001 10001 10001",
            "11111 10000 10000 11110 10000 10000 11111",
            "01110 10001 10001 11111 10001 10001 10001",
            "11110 10001 10001 11110 10000 10000 10000",
            "10000 10000 10000 10000 10000 10000 11111",
            "00000 00000 00000 00000 00000 00000 00000",
        ],
        strict=True,
    )
)


def _palette() -> bytes:
    values = []
    for i in range(240):
        v = i / 239 * 4
        j = min(3, int(v))
        f = v - j
        values.extend(math.floor(z + f * (PALETTE_STOPS[j + 1][k] - z) + 0.5) for k, z in enumerate(PALETTE_STOPS[j]))
    for rgb in (
        (16, 31, 48),
        (255, 255, 255),
        (249, 251, 253),
        (142, 153, 165),
        (88, 116, 142),
        (225, 232, 236),
        (39, 60, 77),
        (210, 220, 230),
    ):
        values.extend(rgb)
    return bytes(values + [0] * 24)


def _label(pixels: bytearray, text: str, x: int, y: int, scale: int = 2) -> None:
    for char in text.upper():
        for row, bits in enumerate(GLYPHS[char].split()):
            for col, active in enumerate(bits):
                if active == "1":
                    for yy in range(scale):
                        for xx in range(scale):
                            px, py = x + col * scale + xx, y + row * scale + yy
                            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                                pixels[py * WIDTH + px] = 240
        x += 6 * scale


def _frame(field: np.ndarray, time_s: int) -> bytes:
    pixels = bytearray([242]) * (WIDTH * HEIGHT)
    _label(pixels, "STEEL PLATE", 13, 9)
    _label(pixels, f"T={time_s}S", 190, 9)
    left, top, cw, ch = 16, 36, 3, 4
    for row in range(GRID_POINTS):
        for col in range(GRID_POINTS):
            level = (field[row, col] - AMBIENT_TEMPERATURE_C) / HOT_SPOT_RISE_C
            colour = max(0, min(239, math.floor(level * 239 + 0.5)))
            for k in range(ch):
                offset = (top + row * ch + k) * WIDTH + left + col * cw
                pixels[offset : offset + cw] = bytes([colour]) * cw
    for col in range(left - 1, left + GRID_POINTS * cw + 1):
        pixels[(top - 1) * WIDTH + col] = 243
        pixels[(top + GRID_POINTS * ch) * WIDTH + col] = 243
    for row in range(top - 1, top + GRID_POINTS * ch + 1):
        pixels[row * WIDTH + left - 1] = pixels[row * WIDTH + left + GRID_POINTS * cw] = 243
    for row in range(top, top + GRID_POINTS * ch):
        level = math.floor((top + GRID_POINTS * ch - 1 - row) / (GRID_POINTS * ch - 1) * 239 + 0.5)
        pixels[row * WIDTH + 275 : row * WIDTH + 288] = bytes([level]) * 13
    _label(pixels, "80", 276, 26, 1)
    _label(pixels, "20", 276, 363, 1)
    return bytes(pixels)


def _compress(data: bytes) -> bytes:
    """Encode palette indices with GIF's least-significant-bit-first LZW."""
    output = bytearray()
    buffer, used = 0, 0

    def emit(code: int, width: int) -> None:
        nonlocal buffer, used
        buffer |= code << used
        used += width
        while used >= BITS_PER_BYTE:
            output.append(buffer & 255)
            buffer >>= 8
            used -= 8

    table, next_code, width = {}, 258, 9
    emit(256, width)
    previous = data[0]
    for current in data[1:]:
        key = (previous << 8) | current
        match = table.get(key)
        if match is not None:
            previous = match
            continue
        emit(previous, width)
        if next_code < MAX_CODES:
            table[key] = next_code
            next_code += 1
            if next_code > (1 << width) and width < MAX_CODE_WIDTH:
                width += 1
        else:
            emit(256, width)
            table, next_code, width = {}, 258, 9
        previous = current
    emit(previous, width)
    emit(257, width)
    if used:
        output.append(buffer & 255)
    return bytes(output)


def simulate_fields(*, frame_count: int = FRAME_COUNT, steps_per_frame: int = STEPS_PER_FRAME) -> list[np.ndarray]:
    """Collect temperature snapshots; every forward-Euler step is evaluated by Bilt and GT4Py."""
    if frame_count < 1 or steps_per_frame < 1:
        msg = "frame_count and steps_per_frame must both be positive"
        raise ValueError(msg)

    temperature = initial_temperature()
    frames = [temperature.copy()]
    ambient = Quantity(AMBIENT_TEMPERATURE_C, absolute_celsius)
    spacing_x, spacing_y = DX_M * meter, DY_M * meter
    diffusivity = STEEL_DIFFUSIVITY_M2_PER_S * meter**2 / second
    time_step = TIME_STEP_S * second

    # Python advances the simulation clock; Bilt owns the heat equation and Euler update.
    with plate_implementation:
        for _ in range(frame_count - 1):
            for _ in range(steps_per_frame):
                plate = HeatPlate(
                    temperature=tensor(temperature, over=(y, x), unit=absolute_celsius),
                    ambient_temperature=ambient,
                    dx=spacing_x,
                    dy=spacing_y,
                    thermal_diffusivity=diffusivity,
                    time_step=time_step,
                )
                temperature = plate.next_temperature.convert_to(absolute_celsius).values
            frames.append(temperature.copy())

    return frames


def generate_gif() -> bytes:
    """Render Bilt/GT4Py output from 0 to 160 seconds, sampled every 20 seconds."""
    fields = simulate_fields()
    output = bytearray(b"GIF89a")
    output.extend(struct.pack("<HHBBB", WIDTH, HEIGHT, 0xF7, 242, 0))
    output.extend(_palette())
    output.extend(bytes.fromhex("21 ff 0b") + b"NETSCAPE2.0" + bytes.fromhex("03 01 00 00 00"))
    for step, field in enumerate(fields):
        data = _compress(_frame(field, int(step * STEPS_PER_FRAME * TIME_STEP_S)))
        output.extend(bytes.fromhex("21 f9 04 00"))
        output.extend(struct.pack("<H", 55))
        output.extend(bytes.fromhex("00 00 2c"))
        output.extend(struct.pack("<HHHHB", 0, 0, WIDTH, HEIGHT, 0))
        output.append(8)
        for pos in range(0, len(data), 255):
            block = data[pos : pos + 255]
            output.append(len(block))
            output.extend(block)
        output.append(0)
    output.append(0x3B)
    return bytes(output)


def main() -> None:
    """Write the animation, or check that the committed GIF is up to date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed GIF differs")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "docs" / "steel-plate.gif"
    content = generate_gif()
    if args.check:
        if not path.exists() or path.read_bytes() != content:
            parser.error("docs/steel-plate.gif differs; regenerate with this script")
    else:
        path.write_bytes(content)


if __name__ == "__main__":
    main()
