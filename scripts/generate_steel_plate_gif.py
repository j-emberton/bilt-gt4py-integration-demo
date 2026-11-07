"""Regenerate the steel-plate animation from an independent periodic heat solver.

Run `uv run python -m scripts.generate_steel_plate_gif` (`--check` to verify).
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np

from steel_plate_example import (
    AMBIENT_TEMPERATURE_C,
    DX_M,
    DY_M,
    GRID_POINTS,
    HOT_SPOT_RISE_C,
    HOT_SPOT_WIDTH_M,
    STEEL_DIFFUSIVITY_M2_PER_S,
    TIME_STEP_S,
)

WIDTH, HEIGHT = 312, 396
PALETTE_STOPS = ((19, 36, 83), (27, 101, 163), (45, 181, 191), (248, 212, 84), (190, 50, 44))
GLYPHS = dict(zip("0123456789TS=CHEAPL ", [
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
], strict=True))


def _palette() -> bytes:
    values = []
    for i in range(240):
        v = i / 239 * 4
        j = min(3, int(v))
        f = v - j
        values.extend(math.floor(z + f * (PALETTE_STOPS[j + 1][k] - z) + 0.5) for k, z in enumerate(PALETTE_STOPS[j]))
    for rgb in ((16, 31, 48), (255, 255, 255), (249, 251, 253), (142, 153, 165), (88, 116, 142), (225, 232, 236), (39, 60, 77), (210, 220, 230)):
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
        while used >= 8:
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
        if next_code < 4096:
            table[key] = next_code
            next_code += 1
            if next_code > (1 << width) and width < 12:
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


def generate_gif() -> bytes:
    """Encode the original 0–160 s animation using an independent reference solver."""
    coordinates_x = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DX_M
    coordinates_y = (np.arange(GRID_POINTS) - GRID_POINTS // 2) * DY_M
    field = np.array([
        [AMBIENT_TEMPERATURE_C + HOT_SPOT_RISE_C * math.exp(-(x * x + y * y) / (2 * HOT_SPOT_WIDTH_M**2)) for x in coordinates_x]
        for y in coordinates_y
    ])
    cx = STEEL_DIFFUSIVITY_M2_PER_S * TIME_STEP_S / DX_M**2
    cy = STEEL_DIFFUSIVITY_M2_PER_S * TIME_STEP_S / DY_M**2
    output = bytearray(b"GIF89a")
    output.extend(struct.pack("<HHBBB", WIDTH, HEIGHT, 0xF7, 242, 0))
    output.extend(_palette())
    output.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
    for step in range(9):
        if step:
            for _ in range(40):
                delta_x = np.roll(field, 1, axis=1) - 2 * field + np.roll(field, -1, axis=1)
                delta_y = np.roll(field, 1, axis=0) - 2 * field + np.roll(field, -1, axis=0)
                field = field + cx * delta_x + cy * delta_y
        data = _compress(_frame(field, step * 20))
        output.extend(b"\x21\xf9\x04\x00")
        output.extend(struct.pack("<H", 55))
        output.extend(b"\x00\x00\x2c")
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
