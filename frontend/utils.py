from io import BytesIO

from PIL import Image, ImageDraw


def image_to_byte_array(image: Image.Image) -> bytes:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="png")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def draw_rectangle(image: Image.Image, bbox: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        ((bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"])),
        outline="Red",
        width=2,
    )
    return img
