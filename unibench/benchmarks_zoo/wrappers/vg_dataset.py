import os
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from unibench.common_utils.utils import load_DINO_mask, get_mask_transform
from ...common_utils import DATA_DIR, MASK_DIR


class VgDataset(Dataset):
    def __init__(self, task_type, transform=None, mask_dir=None):
        self.data_dir = Path(DATA_DIR).joinpath('aro').joinpath('images')
        self.mask_dir = Path(MASK_DIR).joinpath('aro')
        if task_type == 'vga':
            self.json_file = Path(DATA_DIR).joinpath('aro').joinpath('visual_genome_attribution.json')
        elif task_type == 'vgr':
            self.json_file = Path(DATA_DIR).joinpath('aro').joinpath('visual_genome_relation.json')

        self.json_data = self.load_json(self.json_file)

        self.transform = transform
        self.mask_transform = get_mask_transform(transform)

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data[idx]

        image_name = item["image_id"] + '.jpg'
        true_caption = item["true_caption"]
        false_caption = item["false_caption"]

        image_path = self.data_dir.joinpath(image_name)
        image = Image.open(image_path).convert("RGB")

        if not self.mask_dir.exists():
            return self.transform(image), true_caption, false_caption, image_name

        # get mask
        edge_path = self.mask_dir.joinpath(f"{item['image_id']}_edges.pkl")
        edge = load_DINO_mask(edge_path, (image.height, image.width, 3))
        rgba = np.concatenate((image, np.expand_dims(edge, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]

        if max(h, w) == w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)

        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]

        image_torch = self.transform(Image.fromarray(rgb))
        mask_torch = self.mask_transform(Image.fromarray(mask * 255))

        return image_torch, true_caption, false_caption, image_name, mask_torch
