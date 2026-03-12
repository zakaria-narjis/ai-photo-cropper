import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

BASE_URL = "http://backend:8080/"


class CompositionAPI:
    @staticmethod
    def one_crop(image: bytes) -> dict:
        url = BASE_URL + "one_crop/"
        try:
            fields = {
                "image": (image.name.split(".")[0], image, image.type),
            }
        except AttributeError:
            fields = {
                "image": ("image", image, "jpg"),
            }
        multipart_data = MultipartEncoder(fields=fields)
        header = {"Content-Type": multipart_data.content_type}
        response = requests.post(url, data=multipart_data, headers=header)
        return response.json()

    @staticmethod
    def multi_crop(images: list) -> dict:
        url = BASE_URL + "multi_crop/"
        try:
            fields = [
                ("images", (f"{image.name.split('.')[0]}_{id(image)}", image, image.type))
                for image in images
            ]
        except AttributeError:
            fields = [
                ("images", (f"image_{id(image)}", image, "jpg")) for image in images
            ]
        multipart_data = MultipartEncoder(fields=fields)
        header = {"Content-Type": multipart_data.content_type}
        response = requests.post(url, data=multipart_data, headers=header)
        return response.json()


class ClipCropAPI:
    @staticmethod
    def crop(image: bytes, query: str) -> dict:
        url = BASE_URL + "clip_crop/"
        multipart_data = MultipartEncoder(
            fields={
                "image": (image.name, image, image.type),
                "query": query,
            }
        )
        header = {"Content-Type": multipart_data.content_type}
        response = requests.post(url, data=multipart_data, headers=header)
        return response.json()
