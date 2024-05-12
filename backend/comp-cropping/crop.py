import torch
import cv2
from CACNet import CACNet
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from io import BytesIO

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224,224)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHT_FILE = "./pretrained_models/best-FLMS_iou.pth"
# model = CACNet(loadweights=False)
# model.load_state_dict(torch.load(weight_file,map_location=device))
# model = model.to(device).eval()
class Cropper:
    def __init__(self,):
        self.device = DEVICE
        self.model = CACNet(loadweights=False)
        self.model.load_state_dict(torch.load(WEIGHT_FILE,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def process_image(self,image):
        im_width, im_height = image.size
        h = IMAGE_SIZE[1]
        w = IMAGE_SIZE[0]
        resized_image = image.resize((w, h), Image.LANCZOS)
        image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])
        im = image_transformer(resized_image)
        return im, im_width, im_height, image

    def predict(self,image):
        logits,kcm,crop = self.model(image, only_classify=False)
        return logits,kcm,crop

    def crop_image(self,image):
        im, im_width, im_height, source_img = self.process_image(image)
        if im.dim()<4:
            im = im.unsqueeze(0)
        im = im.to(self.device)
        logits,kcm,crop = self.predict(im)
        crop[:,0::2] = crop[:,0::2] / im.shape[-1] * im_width
        crop[:,1::2] = crop[:,1::2] / im.shape[-2] * im_height
        pred_crop = crop.detach().cpu()
        pred_crop[:,0::2] = torch.clip(pred_crop[:,0::2], min=0, max=im_width)
        pred_crop[:,1::2] = torch.clip(pred_crop[:,1::2], min=0, max=im_height)
        pred_crop = pred_crop[0].numpy().tolist()
        x1,y1,x2,y2 = [int(x) for x in pred_crop]
        # cropped_image = cv2.rectangle(np.array(source_img) , (x1,y1), (x2,y2), (255,0,0), 2) 
        # res, cropped_image = cv2.imencode(".jpeg", cv2.cvtColor(cropped_image , cv2.COLOR_BGR2RGB))
        return x1,y1,x2,y2 



class Cropper2:
    def __init__(self,):
        self.device = DEVICE
        self.model = CACNet(loadweights=False)
        self.model.load_state_dict(torch.load(WEIGHT_FILE,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])


    def resize_image(self,image):
        im_width, im_height = image.size
        h = IMAGE_SIZE[1]
        w = IMAGE_SIZE[0]
        resized_image = image.resize((w, h), Image.LANCZOS)
        return  resized_image, im_width, im_height
    
    def process_images(self,images_as_bytes:list[bytes])->tuple[list,list,list]:
        im_widths = []
        im_heights= []
        resized_images =[]     
        for image in images_as_bytes:
            image = Image.open(BytesIO(image)).convert('RGB')
            resized_image, im_width, im_height = self.resize_image(image)
            resized_image = self.image_transformer(resized_image)
            im_widths.append(im_width)
            im_heights.append(im_height)
            resized_images.append(resized_image.unsqueeze(0))
        im_widths = torch.tensor(im_width)
        im_heights = torch.tensor(im_heights)
        resized_images = torch.cat(resized_images)
        return resized_images, im_widths ,im_heights 
    
    def predict(self,image):
        logits,kcm,crop = self.model(image, only_classify=False)
        return logits,kcm,crop

    def crop_images(self,images,multi=False):
        ims, im_widths, im_heights  = self.process_images(images)
        ims = ims.to(self.device)
        logits,kcm,crop = self.predict(ims)
        crop[:,0::2] = crop[:,0::2] / IMAGE_SIZE[0] * im_widths
        crop[:,1::2] = crop[:,1::2] / IMAGE_SIZE[1] * im_heights
        pred_crop = crop.detach().cpu()
        pred_crop = pred_crop.t()
        pred_crop[0::2,:] = torch.clip(pred_crop[0::2,:], min=torch.zeros(pred_crop.shape[1]), max=im_widths)
        pred_crop[1::2,:] = torch.clip(pred_crop[1::2,:], min=torch.zeros(pred_crop.shape[1]), max=im_heights)
        pred_crop = pred_crop.t()
        pred_crop = pred_crop.to(torch.int16)
        if multi:
            return pred_crop.tolist()
        else:
            x1,y1,x2,y2 = [int(x) for x in pred_crop[0].tolist()]
            return  x1,y1,x2,y2 