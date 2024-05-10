from fastapi import FastAPI
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile ,Form
from fastapi.responses import HTMLResponse
from crop import Cropper
from fastapi.responses import Response
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from pydantic import BaseModel
from typing import Annotated
import torch
import torchvision


app = FastAPI()
crop = Cropper()

class Bbox(BaseModel):
    x1:int
    y1:int
    x2:int
    y2:int

class MultiBbox(BaseModel):
    crops:list[Bbox]

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/one_crop/", response_model=Bbox)

# image: Annotated[UploadFile, File(description="Multiple files as UploadFile")], typo: str = Form(...)
async def one_image_crop(image: Annotated[UploadFile, File(description="Upload one image file")]):
    print(image.filename,image.content_type)
    content = await image.read()
    image = Image.open(BytesIO(content)).convert('RGB')
    x1,y1,x2,y2 = crop.crop_image(image)
            # 'image':StreamingResponse(BytesIO(cropped_image.tobytes())),
    response_data = {
        'x1':x1,
        'y1':y1,
        'x2':x2,
        'y2':y2
    }
    return response_data

# @app.post("/multicrop/", response_model = MultiBbox)

# async def multi_image_crop(images: Annoted[])