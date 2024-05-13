from PIL import Image, ImageDraw 
from io import BytesIO


def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format='png')
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def draw_rectangle(image: Image,bbox:dict)->Image:
  img = image.copy()
  draw = ImageDraw.Draw(img)
  draw.rectangle(((bbox["x1"], bbox["y1"]),(bbox["x2"], bbox["y2"])), outline='Red')
  return img