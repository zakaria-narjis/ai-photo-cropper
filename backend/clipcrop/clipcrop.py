from io import BytesIO

import clip
import torch
from PIL import Image

from comp_cropping.crop import ClearCache

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipCrop:
    def __init__(self):
        self.device = DEVICE
        self.yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        self.clip_model.eval()

    def crop(self, image: bytes, search_query: str):
        with ClearCache():
            with torch.no_grad():
                source_img = Image.open(BytesIO(image)).convert("RGB")
                crop_results = self.yolo_model(source_img)
                results = crop_results.crop(save=False)
                preprocessed_images = torch.stack(
                    [self.preprocess(Image.fromarray(r["im"])) for r in results]
                ).to(self.device)
                images_features = self.clip_model.encode_image(preprocessed_images)
                text_encoded = self.clip_model.encode_text(
                    clip.tokenize(search_query).to(self.device)
                )
                images_features /= images_features.norm(dim=-1, keepdim=True)
                text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
                similarity = text_encoded.cpu().numpy() @ images_features.cpu().numpy().T
                x1, y1, x2, y2 = list(map(int, results[similarity.argmax()]["box"]))
                del (
                    similarity,
                    preprocessed_images,
                    search_query,
                    results,
                    crop_results,
                    source_img,
                    images_features,
                    text_encoded,
                )
                return x1, y1, x2, y2
