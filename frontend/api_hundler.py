import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

BASE_URL = 'http://backend:8080/'
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
        return response.json()
    
    def multi_crop(images:list[bytes])->dict:
        url = BASE_URL +'multi_crop/'

        fields=[
                   ('images',(f'{id(image)}_{image.name}',image,image.type)) for image in images
                ]
            
        multipart_data = MultipartEncoder(
            fields = fields
            )
        
        header ={
            'Content-Type': multipart_data.content_type
            }   
        response = requests.post(url, 
                                 data=multipart_data,
                                 headers=header)
        return response.json() 
