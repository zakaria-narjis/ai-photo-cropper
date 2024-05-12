from fastapi import FastAPI
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile ,Form
from fastapi.responses import HTMLResponse
from crop import Cropper, Cropper2
from fastapi.responses import Response
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from pydantic import BaseModel, conlist
from typing import Annotated
from annotated_types import Len
import torch
import torchvision
import json

app = FastAPI()
crop = Cropper2()

class Bbox(BaseModel):
    x1:int
    y1:int
    x2:int
    y2:int

class Crop(BaseModel):
    id : str
    image_name:str
    coords :Bbox

class MultiCrop(BaseModel):
    # crops:list[Annotated[list[int], Len(min_length=4, max_length=4)]]
    crops:list[Crop]


class Image(BaseModel):
    id:str
    file:Annotated[list[UploadFile], File(description="Multiple images files as UploadFile")] 

class MultiCropInput(BaseModel):
    images: Annotated[list[UploadFile], File(description="Multiple images files as UploadFile")] 


@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/one_crop/", response_model=Bbox)
# image: Annotated[UploadFile, File(description="Multiple files as UploadFile")], typo: str = Form(...)
async def one_image_crop(image: Annotated[UploadFile, File(description="One image files as UploadFile")]):
    content = [await image.read()]
    # image = Image.open(BytesIO(content)).convert('RGB')
    x1,y1,x2,y2 = crop.crop_images(images= content, multi = False)
            # 'image':StreamingResponse(BytesIO(cropped_image.tobytes())),
    response_data = {
        'x1':x1,
        'y1':y1,
        'x2':x2,
        'y2':y2
    }
    return response_data

@app.post("/multicrop/", response_model = MultiCrop)
async def multi_image_crop(images: MultiCropInput):
    content = [ await image.read() for image in images]
    crops_list = crop.crop_images(images = content, multi = True)
    response_data = {
        'crops': [{'id':str(id(image)), 
                   'image_name':image.filename, 
                   'coords':{
                        'x1':crop[0],
                        'y1':crop[1],
                        'x2':crop[2],
                        'y2':crop[3]
                    }} for crop,image in zip(crops_list,images)]
                    }
    return response_data
