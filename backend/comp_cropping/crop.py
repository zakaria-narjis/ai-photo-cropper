import torch
from comp_cropping.CACNet import CACNet
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
from torch.utils.data import DataLoader, Dataset


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_FILE = "comp_cropping/pretrained_models/best-FLMS_iou.pth"


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


class ImagesDataset(Dataset):
    def __init__(self, images_as_bytes: list[bytes]):
        self.images_as_bytes = images_as_bytes
        self.IMAGE_SIZE = torch.tensor(IMAGE_SIZE, requires_grad=False)
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])

    def process_images(self, image):
        image = Image.open(BytesIO(image)).convert("RGB")
        im_width, im_height = image.size
        h = self.IMAGE_SIZE[0]
        w = self.IMAGE_SIZE[1]
        resized_image = image.resize((w, h), Image.LANCZOS)
        im_width = torch.tensor([im_width], requires_grad=False)
        im_height = torch.tensor([im_height], requires_grad=False)
        resized_image = self.image_transformer(resized_image)
        resized_image.requires_grad = False
        return resized_image, im_width, im_height

    def __len__(self):
        return len(self.images_as_bytes)

    def __getitem__(self, index):
        return self.process_images(self.images_as_bytes[index])


class Cropper:
    def __init__(self):
        self.device = DEVICE
        self.model = CACNet(loadweights=False)
        self.model.load_state_dict(torch.load(WEIGHT_FILE, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.IMAGE_SIZE = torch.tensor(IMAGE_SIZE, requires_grad=False).to(self.device)

    def predict(self, image):
        with ClearCache():
            with torch.no_grad():
                logits, kcm, crop = self.model(image, only_classify=False)
        return logits, kcm, crop

    def crop_images(self, images: list[bytes], multi=False):
        with ClearCache():
            with torch.no_grad():
                dataset = ImagesDataset(images)
                batch_size = 20
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    drop_last=False,
                )
                crops = []
                for im, im_width, im_height in data_loader:
                    im = im.to(self.device)
                    im_height = torch.reshape(im_height, (1, -1)).to(self.device)
                    im_width = torch.reshape(im_width, (1, -1)).to(self.device)
                    logits, kcm, crop = self.predict(im)
                    crop[:, 0::2] = crop[:, 0::2] / self.IMAGE_SIZE[1] * (im_width.t())
                    crop[:, 1::2] = crop[:, 1::2] / self.IMAGE_SIZE[0] * (im_height.t())
                    pred_crop = crop.t()

                    # Clip the out-of-range bbox coordinates
                    minimum_bbox_value = torch.zeros(im.shape[0]).to(self.device)
                    pred_crop[0::2, :] = torch.clip(pred_crop[0::2, :], min=minimum_bbox_value, max=im_width)
                    pred_crop[1::2, :] = torch.clip(pred_crop[1::2, :], min=minimum_bbox_value, max=im_height)
                    pred_crop = pred_crop.t()
                    pred_crop = pred_crop.to(torch.int16)
                    pred_crop = pred_crop.detach().cpu()
                    del crop, kcm, logits, minimum_bbox_value, im, im_width, im_height
                    crops.extend(pred_crop.tolist())

                del data_loader, dataset
            if multi:
                return crops
            else:
                x1, y1, x2, y2 = [int(x) for x in crops[0]]
                return x1, y1, x2, y2
