from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from clipcrop.clipcrop import ClipCrop
from comp_cropping.crop import ClearCache, Cropper

app = FastAPI()
cropper = Cropper()
clipcrop = ClipCrop()

print(f"Working with {cropper.device}")


class Bbox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Crop(BaseModel):
    image_name: str
    coords: Bbox


class MultiCrop(BaseModel):
    crops: list[Crop]


@app.post("/one_crop/", response_model=Bbox)
async def one_image_crop(
    image: Annotated[UploadFile, File(description="One image file as UploadFile")],
):
    content = [await image.read()]
    x1, y1, x2, y2 = cropper.crop_images(images=content, multi=False)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.post("/multi_crop/", response_model=MultiCrop)
async def multi_image_crop(
    images: Annotated[
        list[UploadFile], File(description="Multiple images files as UploadFile")
    ],
):
    with ClearCache():
        content = [await image.read() for image in images]
        crops_list = cropper.crop_images(images=content, multi=True)
        response_data = {
            "crops": [
                {
                    "image_name": image.filename,
                    "coords": {
                        "x1": c[0],
                        "y1": c[1],
                        "x2": c[2],
                        "y2": c[3],
                    },
                }
                for c, image in zip(crops_list, images)
            ]
        }
        return response_data


@app.post("/clip_crop/", response_model=Bbox)
async def clip_crop(
    image: Annotated[UploadFile, File(description="One image file as UploadFile")],
    query: str = Form(...),
):
    with ClearCache():
        content = await image.read()
        x1, y1, x2, y2 = clipcrop.crop(content, query)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.get("/")
def home():
    return {"health_check": "OK"}
