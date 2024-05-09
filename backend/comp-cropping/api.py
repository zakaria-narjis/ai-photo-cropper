from fastapi import FastAPI
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from crop import Cropper
from fastapi.responses import Response
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from pydantic import BaseModel


app = FastAPI()
crop = Cropper()

class Box(BaseModel):
    coordinates:list[int]

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/uploadfile/", response_model=Box)
async def one_image_crop(images: UploadFile):
    contents = await images.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    _,x1,y1,x2,y2 = crop.crop_image(image)
            # 'image':StreamingResponse(BytesIO(cropped_image.tobytes())),
    response_data = {
        'coordinates':[x1,y1,x2,y2]
    }
    return response_data