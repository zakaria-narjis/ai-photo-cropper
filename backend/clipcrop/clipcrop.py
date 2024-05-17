from PIL import Image
import torch
import clip
import numpy as np
from io import BytesIO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

class ClipCrop:
    def __init__(self,):
        self.device = DEVICE
        self.yolo_model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
        self.clip_model , self.preprocess = clip.load('ViT-B/32',self.device)
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        self.clip_model.eval()

    def crop(self, image, search_query):
        with ClearCache():
            with torch.no_grad(): 
                source_img = Image.open(BytesIO(image)).convert('RGB')
                crop_results = self.yolo_model(source_img)
                results = crop_results.crop(save=False)
                preprocessed_images = torch.stack([
                    self.preprocess(Image.fromarray(result['im'])) for result in results])
                preprocessed_images = torch.tensor(np.stack(preprocessed_images)).to(self.device)            
                images_features = self.clip_model.encode_image(preprocessed_images)
                text_encoded = self.clip_model.encode_text(clip.tokenize(search_query).to(self.device)) 
                images_features /= images_features.norm(dim=-1, keepdim=True)
                text_encoded /= text_encoded.norm(dim=-1, keepdim=True) 
                similarity = text_encoded.cpu().numpy() @ images_features.cpu().numpy().T
                x1,y1,x2,y2= list(map(lambda x:int(x),results[similarity.argmax()]['box']))
                del similarity, preprocessed_images, search_query, 
                results,crop_results, source_img, images_features,text_encoded
                return x1,y1,x2,y2

        