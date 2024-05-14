from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile ,Form
from comp_cropping.crop import Cropper, Cropper2
from pydantic import BaseModel, conlist
from typing import Annotated


app = FastAPI()
crop = Cropper2()
print(f'Working wiht {crop.device}')
class Bbox(BaseModel):
    x1:int
    y1:int
    x2:int
    y2:int

class Crop(BaseModel):
    image_name:str
    coords :Bbox

class MultiCrop(BaseModel):
    crops:list[Crop]

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/one_crop/", response_model=Bbox)

async def one_image_crop(image: Annotated[UploadFile, File(description="One image file as UploadFile")] ):
    content = [await image.read()]

    x1,y1,x2,y2 = crop.crop_images(images= content, multi = False)

    response_data = {
        'x1':x1,
        'y1':y1,
        'x2':x2,
        'y2':y2
    }
    return response_data

@app.post("/multi_crop/", response_model = MultiCrop)
async def multi_image_crop(images: Annotated[list[UploadFile], File(description="Multiple images files as UploadFile")] ):
    content = [ await image.read() for image in images]
    crops_list = crop.crop_images(images = content, multi = True)
    response_data = {
        'crops': [{
                    'image_name':image.filename, 
                   'coords':{
                        'x1':crop[0],
                        'y1':crop[1],
                        'x2':crop[2],
                        'y2':crop[3]
                    }} for crop,image in zip(crops_list,images)]
                    }
    return response_data
