import os
import json

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from unibench.common_utils.utils import load_mask, get_mask_transform


class VgDataset(Dataset):
    def __init__(self, image_dir, json_file, transform=None, mask_dir=None):
        self.image_dir = image_dir
        self.data = self.load_json(json_file)
        self.transform = transform
        self.mask_transform = get_mask_transform(transform)
        self.mask_dir = mask_dir

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_name = item["image_id"] + '.jpg'
        true_caption = item["true_caption"]
        false_caption = item["false_caption"]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # get mask
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, item["image_id"] + '_mask.pkl')
            mask = load_mask(mask_path, (image.height, image.width, 3))
            rgba = np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=-1)
            mask = rgba[:, :, -1]

        image = self.transform(image) if self.transform else image
        mask = self.mask_transform(mask * 255) if mask is not None else None

        if mask is not None:
            return image, mask, true_caption, false_caption, image_name
        else:
            return image, true_caption, false_caption, image_name
