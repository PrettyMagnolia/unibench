"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from PIL import Image
from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import Dataset
from ...common_utils import DATA_DIR, DS_CACHE_DIR, MASK_DIR, load_DINO_mask, get_mask_transform
from pathlib import Path


class HuggingFaceDataset(Dataset):
    def load_txt_file(self, dataset_url, filename, dir):
        file = hf_hub_download(
            repo_id=dataset_url,
            filename=filename,
            repo_type="dataset",
            local_dir=dir,
            local_dir_use_symlinks=False,
            cache_dir=DS_CACHE_DIR
        )

        res = []
        with open(file) as f:
            for line in f:
                res.append(
                    line.replace("_", " ")
                    .replace("\n", "")
                    .replace("{c}", "{}")
                    .replace("  ", " ")
                    .lower()
                )
        return res

    def __init__(
            self,
            dataset_url,
            root: str = DATA_DIR,
            transform=None,
            target_transform=None,
            download_num_workers=60,
            image_extension="webp",
            classes=None,
            templates=None,
            *args,
            **kwargs
    ):
        Dataset.__init__(self)
        assert dataset_url != "", "Please provide a dataset url"

        self.dataset_name = dataset_url.split("/")[-1]
        self.root_dir = root
        self.dataset_dir = Path(self.root_dir) / self.dataset_name
        self.dataset_url = dataset_url
        self.image_extension = image_extension
        self.transform = transform
        self.download_num_workers = download_num_workers
        self.target_transform = target_transform

        self.classes = classes
        self.templates = templates

        if not self.dataset_dir.exists():
            self.download_dataset()

        self.dataset = load_from_disk(str(self.dataset_dir))

        self.mask_dir = MASK_DIR.joinpath(self.dataset_name)
        self.mask_transform = get_mask_transform(transform)

        try:
            if classes is None:
                self.classes = self.load_txt_file(
                    dataset_url, "classnames.txt", str(self.dataset_dir)
                )
            if templates is None:
                self.templates = self.load_txt_file(
                    dataset_url,
                    "zeroshot_classification_templates.txt",
                    str(self.dataset_dir)
                )
        except:
            pass

    def __len__(self):
        return len(self.dataset)

    def download_dataset(self):
        try:
            self.dataset = load_dataset(
                self.dataset_url,
                trust_remote_code=True,
                split="test",
                num_proc=self.download_num_workers,
                cache_dir=DS_CACHE_DIR
            )
        except:
            self.dataset = load_dataset(
                self.dataset_url,
                trust_remote_code=True,
                split="test",
                cache_dir=DS_CACHE_DIR
            )

        self.dataset.save_to_disk(str(self.dataset_dir))

    def __getitem__(self, index):
        item = self.dataset[index]

        # Loading Images
        samples = []
        for k in item.keys():
            if self.image_extension in k or 'jpg' in k or 'jpeg' in k or 'png' in k:
                img = item[k].convert("RGB")
                copy_img = img
                if self.transform is not None:
                    img = self.transform(img)
                samples.append(img)

        if len(samples) == 1:
            samples = samples[0]

        # Loading Labels
        if "cls" in item.keys():
            target = item["cls"]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = item["npy"]
            if all(isinstance(t, int) for t in target):
                target = torch.nn.functional.one_hot(
                    torch.tensor(target), len(self.classes)
                ).sum(0)

            for t in target:
                if self.target_transform is not None:
                    t = self.target_transform(t)

        if "split.txt" in item.keys():
            return (
                samples,
                target,
                str(item["__key__"]),
                # item["split.txt"].decode("utf-8"),
                item["split.txt"],
            )
        if not self.mask_dir.exists():
            return samples, target, str(item["__key__"])

        # get mask
        edge_path = self.mask_dir.joinpath(f"{item['__key__']}_edges.pkl")
        edge = load_DINO_mask(edge_path, (copy_img.height, copy_img.width, 3))
        rgba = np.concatenate((copy_img, np.expand_dims(edge, axis=-1)), axis=-1)
        mask = rgba[:, :, -1]

        mask_torch = self.mask_transform(Image.fromarray(mask * 255))

        return samples, target, str(item["__key__"]), mask_torch
