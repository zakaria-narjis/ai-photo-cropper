import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
from io import BytesIO

BASE_URL = 'http://127.0.0.1:8000/'
class CompositionAPI:

    def one_crop(image:bytes)->dict:
        url = BASE_URL +'one_crop/'
        multipart_data = MultipartEncoder(
            fields={
               'image':(image.name,image,image.type),
            }
            )
        header ={
            'Content-Type': multipart_data.content_type
            }
        
        response = requests.post(url, 
                                 data=multipart_data,
                                 headers=header)
        print(response.json())
        return response.json()
