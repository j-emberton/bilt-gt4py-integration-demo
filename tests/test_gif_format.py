"""The committed animation must have valid GIF block headers, not escaped text."""

import struct
from pathlib import Path


def test_animation_gif_has_binary_extension_headers() -> None:
    image = Path(__file__).resolve().parents[1] / "docs" / "steel-plate.gif"
    gif = image.read_bytes()

    assert gif[:6] == b"GIF89a"
    width, height, packed = struct.unpack_from("<HHB", gif, 6)
    assert (width, height) == (312, 396)

    colour_table_size = 3 * (1 << ((packed & 0x07) + 1)) if packed & 0x80 else 0
    start = 13 + colour_table_size
    assert gif[start : start + 3] == bytes.fromhex("21 ff 0b")
    assert gif[start + 3 : start + 14] == b"NETSCAPE2.0"
    assert gif.endswith(b";")
