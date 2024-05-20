import torch
from comp_cropping.CACNet import CACNet
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
from torch.utils.data import DataLoader, Dataset
import subprocess as sp
from torch.profiler import profile, record_function, ProfilerActivity


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224,224)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_INDEX = 0
WEIGHT_FILE = "comp_cropping/pretrained_models/best-FLMS_iou.pth"

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

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
        self.IMAGE_SIZE = torch.tensor(IMAGE_SIZE, requires_grad=False).to(self.device)


    def resize_image(self,image):
        im_width, im_height = image.size
        h = self.IMAGE_SIZE[0]
        w = self.IMAGE_SIZE[1]
        resized_image = image.resize((w, h), Image.LANCZOS)
        return  resized_image, im_width, im_height
    
    def process_images(self,images_as_bytes:list[bytes])->tuple[list,list,list]:
        with ClearCache():
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

            im_widths = torch.tensor(im_widths, requires_grad=False).unsqueeze(0).to(self.device)
            im_heights = torch.tensor(im_heights, requires_grad=False).unsqueeze(0).to(self.device)
            resized_images = torch.cat(resized_images).to(self.device)
            resized_images.requires_grad=False
            return resized_images, im_widths ,im_heights 
    
    def predict(self,image):
        with ClearCache():
            with torch.no_grad():
                logits,kcm,crop = self.model(image, only_classify=False)
            return logits,kcm,crop

    def crop_images(self,images:list[bytes],multi=False):
        with ClearCache():
            with torch.no_grad():
                ims, im_widths, im_heights  = self.process_images(images)
                ims = ims.to(self.device)
                logits,kcm,crop = self.predict(ims)
                print(ims.shape,im_widths.shape,im_heights.shape)
                crop[:,0::2] = crop[:,0::2] / self.IMAGE_SIZE[1] * (im_widths.t())
                crop[:,1::2] = crop[:,1::2] / self.IMAGE_SIZE[0] * (im_heights.t())
                pred_crop = crop.t()
                
                #Clip the out of range bbox crop
                number_of_images = pred_crop.shape[1]
                minimum_bbox_value = torch.zeros(number_of_images).to(self.device)        
                pred_crop[0::2,:] = torch.clip(pred_crop[0::2,:], min=minimum_bbox_value , max=im_widths)
                pred_crop[1::2,:] = torch.clip(pred_crop[1::2,:], min=minimum_bbox_value , max=im_heights)

                pred_crop = pred_crop.t()
                pred_crop = pred_crop.to(torch.int16)
                pred_crop = pred_crop.detach().cpu()
                del crop,kcm,logits,minimum_bbox_value ,ims, im_widths, im_heights 
                if multi:
                    return pred_crop.tolist()
                else:
                    x1,y1,x2,y2 = [int(x) for x in pred_crop[0].tolist()]
                    return  x1,y1,x2,y2 
class ImagesDataset(Dataset):
    def __init__(self, images_as_bytes:list[bytes]) :
        self.images_as_bytes = images_as_bytes 
        self.IMAGE_SIZE = torch.tensor(IMAGE_SIZE, requires_grad=False)
        self.image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    def  process_images(self,image):
        image = Image.open(BytesIO(image)).convert('RGB')
        im_width, im_height = image.size
        h = self.IMAGE_SIZE[0]
        w = self.IMAGE_SIZE[1]
        resized_image = image.resize((w, h), Image.LANCZOS)
        im_width = torch.tensor([im_width], requires_grad=False)
        im_height = torch.tensor([im_height], requires_grad=False)
        resized_image = self.image_transformer(resized_image)
        resized_image.requires_grad=False
        return  resized_image, im_width, im_height  
    
    def __len__(self,):
        return len(self.images_as_bytes)
    
    def __getitem__(self,index):
        return self.process_images(self.images_as_bytes[index])

class Cropper:
    def __init__(self,):
        self.device = DEVICE
        self.model = CACNet(loadweights=False)
        self.model.load_state_dict(torch.load(WEIGHT_FILE,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.IMAGE_SIZE = torch.tensor(IMAGE_SIZE, requires_grad=False).to(self.device)

    def predict(self,image):
        with ClearCache():
            with torch.no_grad():
                logits,kcm,crop = self.model(image, only_classify=False)
        return logits,kcm,crop
            
    # def get_gpu_memory():
    #     command = "nvidia-smi --query-gpu=memory.free --format=csv"
    #     memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    #     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    #     return memory_free_values   
    # def batch_size(self,data_len):
    #     "Compute optimal batch size for the available gpu/cpu memory"
    #     if self.device.type == 'cuda':
    #         available_memory = self.get_gpu_memory()[GPU_INDEX]*1024   #available_memory in bytes
    #         data_size = data_len * IMAGE_SIZE[0]*IMAGE_SIZE[1]*3
    #         memory_fraction = available_memory/data_size
    #         if memory_fraction >1:
    #             return data_len
    #         else:
    #             return int(memory_fraction*data_len)

    def crop_images(self,images:list[bytes],multi=False):
        with ClearCache():           
            with torch.no_grad():
                dataset = ImagesDataset(images)
                batch_size  = 20
                data_loader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False
                                        )
                crops =[]
                for im,im_width,im_height in data_loader:
                    im = im.to(self.device)
                    im_height=torch.reshape(im_height,(1,-1)).to(self.device)
                    im_width=torch.reshape(im_width,(1,-1)).to(self.device)
                    logits,kcm,crop = self.predict(im)
                    crop[:,0::2] = crop[:,0::2] / self.IMAGE_SIZE[1] * (im_width.t())
                    crop[:,1::2] = crop[:,1::2] / self.IMAGE_SIZE[0] * (im_height.t())
                    pred_crop = crop.t()
                
                    #Clip the out of range bbox crop
                    minimum_bbox_value = torch.zeros(im.shape[0]).to(self.device)        
                    pred_crop[0::2,:] = torch.clip(pred_crop[0::2,:], min=minimum_bbox_value , max=im_width)
                    pred_crop[1::2,:] = torch.clip(pred_crop[1::2,:], min=minimum_bbox_value , max=im_height)
                    pred_crop = pred_crop.t()
                    pred_crop = pred_crop.to(torch.int16)
                    pred_crop = pred_crop.detach().cpu()
                    del crop,kcm,logits,minimum_bbox_value ,im, im_width, im_height
                    crops.extend(pred_crop.tolist())

                del data_loader,dataset
            if multi:
                return crops
            else:
                x1,y1,x2,y2 = [int(x) for x in crops[0]]
                return  x1,y1,x2,y2 