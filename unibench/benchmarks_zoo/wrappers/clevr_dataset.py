import os
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from unibench.common_utils.utils import load_DINO_mask, get_mask_transform
import webdataset as wds
import io

class CLEVRDataset(Dataset):
    def __init__(self, transform=None, **kwargs):
        self.data_dir = '/mnt/user_data/wenwen/data/clevr/clevr_basic_100000/webdataset_output'
        self.mask_dir = '/mnt/shared/data/DINO_SAM2_Data/clevr/clevr_basic_100000_RP'

        self.dataset = []
        for data_file in os.listdir(self.data_dir):
            if data_file.endswith('10.tar') or data_file.endswith('11.tar'):
                data_path = os.path.join(self.data_dir, data_file)
                wds_dataset = wds.WebDataset(data_path, handler=wds.warn_and_continue, shardshuffle=False)
                for sample in wds_dataset:
                    edge_path = os.path.join(self.mask_dir, 'CLEVR_sample_' + sample['__key__'] + '_edges.pkl')
                    if os.path.exists(edge_path):
                        self.dataset.append(sample)

        self.transform = transform
        self.mask_transform = get_mask_transform(transform)

        self.templates = ['There are {} objects']
        self.classes = ['one', 'two', 'three', 'four', 'five', 'six', 'seven']

        self.has_mask = kwargs['has_mask'] if kwargs['has_mask'] else False
        self.test_mode = kwargs['test_mode'] if kwargs['test_mode'] else None

    def load_json(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        json_info = json.loads(item['json'])
        
        image_stream = io.BytesIO(item["jpg"])
        image = Image.open(image_stream).convert("RGB")


        objects_nums = len(json_info['objects'])
        target = int(objects_nums) - 1

        if not self.has_mask:
            return self.transform(image), target, str(item["__key__"]), ''

        # get mask
        edge_path = os.path.join(self.mask_dir, 'CLEVR_sample_' + item['__key__'] + '_edges.pkl')
        edge = load_DINO_mask(edge_path, (image.height, image.width, 3), self.test_mode)
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

        return image_torch, target, str(item["__key__"]), '', mask_torch
