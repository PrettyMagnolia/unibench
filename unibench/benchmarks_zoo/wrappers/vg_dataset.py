import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class VgDataset(Dataset):
    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = image_dir
        self.data = self.load_json(json_file)
        self.transform = transform

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 获取图片路径和caption
        image_name = item["image_id"] + '.jpg'
        true_caption = item["true_caption"]  # 正确的caption
        false_caption = item["false_caption"]  # 错误的caption

        # 加载图像
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, true_caption, false_caption, image_name
