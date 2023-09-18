import cv2
import enum
import numpy as np

from PIL import Image
from typing import Optional

class Color(enum.Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def invert(self) -> "Color":
        if self == Color.BLACK:
            return Color.WHITE
        else:
            assert self == Color.WHITE
            return Color.BLACK

def generate_checkerboard(*, width: int, height: int, upscale_width: Optional[int] = None, upscale_height: Optional[int] = None, starting_color: Color = Color.WHITE) -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")

    image = Image.new("RGB", (width, height))
    color_a = starting_color
    color_b = starting_color.invert()

    for y in range(height):
        for x in range(width):
            color = color_a if x % 2 == 0 else color_b
            image.putpixel((x, y), color.value)
        color_a, color_b = color_b, color_a

    if upscale_width is not None and upscale_height is not None:
        if upscale_width < width or upscale_height < height:
            raise ValueError("Upscale width and height must be greater than or equal to width and height.")
        image = image.resize((upscale_width, upscale_height), Image.NEAREST)
    else:
        assert upscale_width is None and upscale_height is None

    return image

def generate_checkerboard_cv2(rows=64, columns=64, width=512, height=512):
    assert isinstance(rows, int) and rows > 0
    assert isinstance(columns, int) and columns > 0
    assert isinstance(width, int) and width >= columns
    assert isinstance(height, int) and height >= rows
    img = generate_checkerboard(width=columns, height=rows, upscale_width=width, upscale_height=height)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img
